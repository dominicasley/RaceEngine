module;

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

export module raceengine.physics:SetupFile;

import :Driveline;
import :Vehicle;

namespace raceengine
{

// A car's setup as a file, so the dozens of numbers that decide how it drives can be changed
// between two laps rather than between two builds.
//
// **It is an overlay and not a car.** What is in here is what a race engineer changes on a Sunday:
// spring and damper rates, bars, brake balance, the differential's ramps, the tyre pressures and the
// force feedback gain. What is *not* in here is the geometry, the masses, the torque curve or the
// hardpoints — those are the car, they come from `PublishedCars`, and a format that could change
// them would be a format in which a typo silently produces a different vehicle and calls it a setup
// change. Every field is therefore optional: absent means "whatever the car says", which is the only
// reading under which a half-written file is a valid one.
//
// Parsing is pure and returns `std::expected`, for the reason `parseCubeLookupTable` is: a text
// format with a grammar is exactly the thing to pin without a device, a disk or a game around it.
//
// `std::from_chars` rather than a stream, and for the same reason `telemetryToCsv` uses `to_chars`:
// a setup file written on a machine with a comma decimal separator is not a setup file, and nothing
// notices until somebody else opens it.

// One axle's worth. Stated per axle rather than per corner because that is how a setup sheet is
// written and how a driver asks for a change — "more front bar", never "more left front bar".
export struct AxleTune
{
    std::optional<double> springRate;
    std::optional<double> bumpRate;
    std::optional<double> reboundRate;
    std::optional<double> antiRollRate;
    std::optional<double> brakeTorque;
};

export struct DifferentialTune
{
    std::optional<double> preload;
    std::optional<double> powerRamp;
    std::optional<double> coastRamp;
};

export struct FeedbackTune
{
    std::optional<double> gain;
    std::optional<double> ceilingTorque;
    // N·m per radian per second of measured rim speed, opposing it — the damper a base with no
    // reachable tuning menu cannot supply for itself. `ForceMapping::damping` explains why it is a
    // torque and not a filter.
    std::optional<double> damping;
    // Hz, the corner of the two-pole low pass on the damper's *measurement* of the rim. It is here
    // because it is the one number that decides how much damping this loop can carry at all, and
    // the answer depends on the base's own delay — so it belongs beside the damping it bounds
    // rather than compiled in. `ForceMapping::damperBandwidth` carries the derivation.
    std::optional<double> damperBandwidth;
};

// Which way the rack moves for a positive steering demand.
//
// Data rather than a code path, and it is here rather than in the input layer because it is a
// property of the *car*: a rack ahead of the axle centreline steers the opposite way to one behind
// it for the same travel, so the answer is per vehicle and belongs with the vehicle's other numbers.
// Inverting in the input layer instead would fix one car and break the next, and would put the force
// feedback out of step with the steering, since stage one derives its torque from where the rack
// actually is.
export struct SteeringTune
{
    std::optional<bool> invert;
};

export struct VehicleTune
{
    AxleTune front;
    AxleTune rear;
    DifferentialTune differential;
    FeedbackTune feedback;
    SteeringTune steering;
};

namespace
{

[[nodiscard]] std::string_view trim(std::string_view text)
{
    const auto blank = [](const char character)
    {
        return character == ' ' || character == '\t' || character == '\r' || character == '\n';
    };

    while (!text.empty() && blank(text.front()))
    {
        text.remove_prefix(1);
    }

    while (!text.empty() && blank(text.back()))
    {
        text.remove_suffix(1);
    }

    return text;
}

// Every quantity in this file is a rate, a torque or a fraction, and not one of them is meaningfully
// negative. Refusing rather than clamping, because a negative spring rate is a typo and a car built
// from it drives itself into the ground in a way that reads as an integrator fault.
[[nodiscard]] std::expected<double, std::string> positive(const std::string_view key, const std::string_view value)
{
    auto parsed = 0.0;
    const auto begin = value.data();
    const auto end = begin + value.size();
    const auto answer = std::from_chars(begin, end, parsed);

    if (answer.ec != std::errc{} || answer.ptr != end)
    {
        return std::unexpected(std::string("'").append(key).append("' is not a number: '").append(value).append("'"));
    }

    if (!(parsed >= 0.0))
    {
        return std::unexpected(std::string("'").append(key).append("' must not be negative"));
    }

    return parsed;
}

} // namespace

// One tune, from the text of a file. Line oriented, `key value`, `#` to the end of the line is a
// comment, blank lines ignored.
//
// **An unknown key is an error rather than a line to skip**, which is the whole difference between a
// setup file and a wish. A misspelled `front.sping` that is quietly ignored is a change the driver
// made, felt nothing from, and then spent the afternoon compensating for elsewhere — the same shape
// as the inert `peakSlipScale` this codebase already caught once.
export [[nodiscard]] std::expected<VehicleTune, std::string> parseVehicleTune(const std::string_view text)
{
    auto tune = VehicleTune{};
    auto lineNumber = std::size_t{0};

    auto remaining = text;
    while (!remaining.empty())
    {
        lineNumber++;

        const auto breakAt = remaining.find('\n');
        auto line = breakAt == std::string_view::npos ? remaining : remaining.substr(0, breakAt);
        remaining = breakAt == std::string_view::npos ? std::string_view{} : remaining.substr(breakAt + 1);

        if (const auto comment = line.find('#'); comment != std::string_view::npos)
        {
            line = line.substr(0, comment);
        }

        line = trim(line);
        if (line.empty())
        {
            continue;
        }

        const auto gap = line.find_first_of(" \t");
        if (gap == std::string_view::npos)
        {
            return std::unexpected(std::string("line ")
                                       .append(std::to_string(lineNumber))
                                       .append(": '")
                                       .append(line)
                                       .append("' has a key and no value"));
        }

        const auto key = line.substr(0, gap);
        const auto value = trim(line.substr(gap));

        if (value.empty())
        {
            return std::unexpected(std::string("line ")
                                       .append(std::to_string(lineNumber))
                                       .append(": '")
                                       .append(key)
                                       .append("' has a key and no value"));
        }

        const auto number = positive(key, value);
        if (!number)
        {
            return std::unexpected(
                std::string("line ").append(std::to_string(lineNumber)).append(": ").append(number.error()));
        }

        const auto assign = [&](AxleTune& axle, const std::string_view field) -> bool
        {
            if (field == "spring")
            {
                axle.springRate = *number;
            }
            else if (field == "bump")
            {
                axle.bumpRate = *number;
            }
            else if (field == "rebound")
            {
                axle.reboundRate = *number;
            }
            else if (field == "antiroll")
            {
                axle.antiRollRate = *number;
            }
            else if (field == "brake")
            {
                axle.brakeTorque = *number;
            }
            else
            {
                return false;
            }

            return true;
        };

        auto known = true;

        if (key.starts_with("front."))
        {
            known = assign(tune.front, key.substr(6));
        }
        else if (key.starts_with("rear."))
        {
            known = assign(tune.rear, key.substr(5));
        }
        else if (key == "diff.preload")
        {
            tune.differential.preload = *number;
        }
        else if (key == "diff.power_ramp")
        {
            tune.differential.powerRamp = *number;
        }
        else if (key == "diff.coast_ramp")
        {
            tune.differential.coastRamp = *number;
        }
        else if (key == "steering.invert")
        {
            tune.steering.invert = *number != 0.0;
        }
        else if (key == "ffb.gain")
        {
            tune.feedback.gain = *number;
        }
        else if (key == "ffb.ceiling")
        {
            tune.feedback.ceilingTorque = *number;
        }
        else if (key == "ffb.damping")
        {
            tune.feedback.damping = *number;
        }
        else if (key == "ffb.damperhz")
        {
            tune.feedback.damperBandwidth = *number;
        }
        else
        {
            known = false;
        }

        if (!known)
        {
            return std::unexpected(std::string("line ")
                                       .append(std::to_string(lineNumber))
                                       .append(": nothing here is called '")
                                       .append(key)
                                       .append("'"));
        }
    }

    return tune;
}

// Whether this sheet says anything at all. A file of nothing but comments is a valid setup and is
// what ships — so a game can tell "the driver changed something" from "there is a setup file", and
// not announce a reload that changed no number on the car.
export [[nodiscard]] bool statesAnything(const VehicleTune& tune)
{
    const auto axle = [](const AxleTune& sheet)
    {
        return sheet.springRate || sheet.bumpRate || sheet.reboundRate || sheet.antiRollRate || sheet.brakeTorque;
    };

    return axle(tune.front) || axle(tune.rear) || tune.differential.preload || tune.differential.powerRamp ||
           tune.differential.coastRamp || tune.feedback.gain || tune.feedback.ceilingTorque || tune.feedback.damping ||
           tune.feedback.damperBandwidth || tune.steering.invert;
}

// The tune onto a car. Absent fields leave what the car said, which is what makes a setup file that
// states one number a setup file that changes one thing.
//
// **The car must be freshly built, not the one already being driven.** An overlay applied on top of
// itself is one that can only ever add: delete a line, save, and the value it used to state is still
// in the car with nothing left in the file to explain it. That is the same failure the unknown-key
// rule exists to prevent — a change the driver made and felt nothing from — and it is also what makes
// `steering.invert` expressible as the flip it actually is rather than as a claim about a sign.
export void applyVehicleTune(const VehicleTune& tune, VehicleSetup& vehicle)
{
    const auto axle = [](const AxleTune& sheet, CornerSetup& corner)
    {
        if (sheet.springRate)
        {
            corner.springRate = *sheet.springRate;
        }

        if (sheet.bumpRate || sheet.reboundRate)
        {
            // The damper is a curve and a linear one is two points, so a sheet stating one of the
            // two has to be read against what the other already is rather than against a default.
            //
            // Bump is the **positive** velocity side, which is `linearDamper`'s convention and is
            // worth reading off it rather than assuming: written the other way round, every damper
            // change in every setup file lands on the stroke it was not meant for, and the car is
            // softer over kerbs when it was asked to be firmer on the brakes.
            const auto bump = sheet.bumpRate.value_or(corner.damper.at(1.0));
            const auto rebound = sheet.reboundRate.value_or(corner.damper.at(-1.0) / -1.0);

            corner.damper = linearDamper(bump, rebound);
        }

        if (sheet.antiRollRate)
        {
            corner.antiRollRate = *sheet.antiRollRate;
        }

        if (sheet.brakeTorque)
        {
            corner.brakeTorque = *sheet.brakeTorque;
        }
    };

    if (tune.steering.invert)
    {
        // Stated by the sign of the travel, the way an inverted axis is stated by its ends. Applying
        // it here means everything downstream — the kinematic solve, the rack torque the wheel is
        // given — follows from one number rather than from a flag each of them has to remember.
        //
        // **This is what the car is actually steered by, and it is the driver's setting.** It was
        // briefly rewritten as a relative flip, on the argument that stating a field's default should
        // be indistinguishable from not stating it — which is true of every other field here and is
        // *not* worth having at the price of changing which way the car goes. `steering.invert 0`
        // means the rack travel is positive, this car is driven with it, and nothing may quietly
        // reinterpret it. See the note in the workspace CLAUDE.md about what `rackTravelForSteer`
        // derives and why it disagrees.
        const auto wanted = *tune.steering.invert;
        const auto inverted = vehicle.rackTravelPerInput < 0.0;

        if (wanted != inverted)
        {
            vehicle.rackTravelPerInput = -vehicle.rackTravelPerInput;
        }
    }

    axle(tune.front, vehicle.corners[0]);
    axle(tune.front, vehicle.corners[1]);
    axle(tune.rear, vehicle.corners[2]);
    axle(tune.rear, vehicle.corners[3]);
}

// And onto the driveline, which owns the one thing on a setup sheet that is not a corner.
export void applyVehicleTune(const VehicleTune& tune, DrivelineSetup& driveline)
{
    const auto& sheet = tune.differential;

    if (!sheet.preload && !sheet.powerRamp && !sheet.coastRamp)
    {
        return;
    }

    auto& pack = driveline.differential;

    pack.preload = sheet.preload.value_or(pack.preload);
    pack.powerRamp = sheet.powerRamp.value_or(pack.powerRamp);
    pack.coastRamp = sheet.coastRamp.value_or(pack.coastRamp);
}

} // namespace raceengine
