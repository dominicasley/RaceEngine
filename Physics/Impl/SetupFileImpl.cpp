// SetupFile bodies. Declarations are in Api/SetupFile.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface partition is part of the module's BMI instead, and editing one rebuilt every importer
// of `raceengine`. Measurements and the rule: docs/build-times.md.
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

module raceengine.physics;

namespace raceengine
{

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

[[nodiscard]] std::expected<VehicleTune, std::string> parseVehicleTune(const std::string_view text)
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

        // **Word-valued keys are read before the value is taken as a number**, because a mode is a
        // name and not a magnitude: `assist.tc 2` is a sheet nobody can read and a typo nobody can
        // see. Every other key on the sheet is a quantity and goes through `positive` below.
        if (key == "assist.tc")
        {
            if (value == "off")
            {
                tune.assists.traction = TractionMode::Off;
            }
            else if (value == "full")
            {
                tune.assists.traction = TractionMode::Full;
            }
            else if (value == "sport")
            {
                tune.assists.traction = TractionMode::Sport;
            }
            else
            {
                return std::unexpected(std::string("line ")
                                           .append(std::to_string(lineNumber))
                                           .append(": 'assist.tc' takes 'off', 'full' or 'sport', not '")
                                           .append(value)
                                           .append("'"));
            }

            continue;
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
            else if (field == "friction")
            {
                axle.damperFriction = *number;
            }
            else if (field == "stopdamping")
            {
                axle.stopDamping = *number;
            }
            else if (field == "stophysteresis")
            {
                axle.stopHysteresis = *number;
            }
            else if (field == "compliancesteer")
            {
                axle.complianceSteer = *number;
            }
            else if (field == "compliancecamber")
            {
                axle.complianceCamber = *number;
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
        else if (key == "assist.abs")
        {
            tune.assists.antilock = *number != 0.0;
        }
        else if (key == "assist.xds")
        {
            tune.assists.cornering = *number != 0.0;
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
        else if (key == "pedal.onset")
        {
            tune.pedal.onsetPeaks = *number;
        }
        else if (key == "pedal.brake_full")
        {
            tune.pedal.brakeFullPeaks = *number;
        }
        else if (key == "pedal.throttle_full")
        {
            tune.pedal.throttleFullPeaks = *number;
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

[[nodiscard]] bool statesAnything(const VehicleTune& tune)
{
    const auto axle = [](const AxleTune& sheet)
    {
        return sheet.springRate || sheet.bumpRate || sheet.reboundRate || sheet.antiRollRate || sheet.brakeTorque ||
               sheet.damperFriction || sheet.stopDamping || sheet.stopHysteresis || sheet.complianceSteer ||
               sheet.complianceCamber;
    };

    return axle(tune.front) || axle(tune.rear) || tune.differential.preload || tune.differential.powerRamp ||
           tune.differential.coastRamp || tune.feedback.gain || tune.feedback.ceilingTorque || tune.feedback.damping ||
           tune.feedback.damperBandwidth || tune.steering.invert || tune.assists.antilock || tune.assists.traction ||
           tune.assists.cornering || tune.pedal.onsetPeaks || tune.pedal.brakeFullPeaks || tune.pedal.throttleFullPeaks;
}

void applyVehicleTune(const VehicleTune& tune, VehicleSetup& vehicle)
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

        if (sheet.damperFriction)
        {
            corner.damperFriction = *sheet.damperFriction;
        }

        // The bump stop only. The droop stop is the damper topping out — a different mechanism —
        // and a sheet key that quietly reached both would be a change the driver did not ask for.
        if (sheet.stopDamping)
        {
            corner.bumpStop.damping = *sheet.stopDamping;
        }

        if (sheet.stopHysteresis)
        {
            corner.bumpStop.hysteresis = *sheet.stopHysteresis;
        }

        if (sheet.complianceSteer)
        {
            // Degrees per kilonewton on the sheet, radians per newton in the model. One conversion,
            // here, because a sheet a driver edits should carry the unit the measurement is published
            // in and the model should carry the unit its arithmetic is in.
            corner.lateralForceSteer = *sheet.complianceSteer * 0.017453292519943295 / 1000.0;
        }

        if (sheet.complianceCamber)
        {
            // The same conversion for the same reason.
            corner.lateralForceCamber = *sheet.complianceCamber * 0.017453292519943295 / 1000.0;
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

void applyVehicleTune(const VehicleTune& tune, AssistSetup& assists)
{
    assists.antilock.enabled = tune.assists.antilock.value_or(assists.antilock.enabled);
    assists.traction.mode = tune.assists.traction.value_or(assists.traction.mode);
    assists.cornering.enabled = tune.assists.cornering.value_or(assists.cornering.enabled);
}

void applyVehicleTune(const VehicleTune& tune, DrivelineSetup& driveline)
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
