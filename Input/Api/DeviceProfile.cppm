module;

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

export module raceengine.input:DeviceProfile;

import :InputBackend;

namespace raceengine
{

// One axis, as this device reads and this driver wants it.
//
// `minimum` and `maximum` are the raw values at the two ends of the *useful* travel, and `minimum`
// above `maximum` is how an inverted axis is stated. There is no separate inversion flag because
// inversion is not a property distinct from which end is which — a flag beside the ends is two
// statements of one fact, and the pair can disagree.
export struct AxisCalibration
{
    double minimum = 0.0;
    double maximum = 1.0;
    // Only read by the bipolar mapping. Measured rather than assumed to be the midpoint: this base
    // rests at 32273 of 0..65535, three quarters of a percent off centre, and a rack fed the
    // difference pulls to one side for ever.
    double centre = 0.5;
    // Fraction of travel ignored at the resting end — either side of `centre` for a bipolar axis.
    // Zero here and zero on purpose: the driver already zeroes fuzz and flat on this device, and a
    // deadzone on a wheel is the thing that makes small corrections feel like nothing at all.
    double deadzone = 0.0;
    // Fraction of travel at the far end that already counts as the maximum. What a pedal needs when
    // its last few millimetres are stiffer than a leg will push.
    double saturation = 0.0;
};

// Where the axis sits between its ends, [0, 1], before any shaping.
export [[nodiscard]] inline double axisTravel(const AxisCalibration& calibration, const double raw)
{
    const auto span = calibration.maximum - calibration.minimum;
    if (span == 0.0)
    {
        return 0.0;
    }

    return std::clamp((raw - calibration.minimum) / span, 0.0, 1.0);
}

// Applies the ends the driver actually uses to a [0, 1] travel: the deadzone is taken off the
// bottom, the saturation off the top, and what is left is stretched back over the whole range so
// that a trimmed axis still reaches one.
export [[nodiscard]] inline double applyEnds(const double travel, const double deadzone, const double saturation)
{
    const auto usable = 1.0 - std::clamp(deadzone, 0.0, 1.0) - std::clamp(saturation, 0.0, 1.0);
    if (usable <= 0.0)
    {
        // Trims that meet or cross leave a step, and the step is at the *deadzone* — which is the
        // limit of the expression below as the usable span closes, and is also the only safe of the
        // two answers: an axis that reads full where it should read nothing is a throttle stuck
        // open. A profile stating such a pair is refused when it is parsed; this is for the one
        // constructed in code.
        return travel >= std::clamp(deadzone, 0.0, 1.0) ? 1.0 : 0.0;
    }

    return std::clamp((travel - deadzone) / usable, 0.0, 1.0);
}

// A pedal: [0, 1], resting at zero.
export [[nodiscard]] inline double normaliseUnipolar(const AxisCalibration& calibration, const double raw)
{
    return applyEnds(axisTravel(calibration, raw), calibration.deadzone, calibration.saturation);
}

// A rim: [-1, 1], centred where the device actually rests.
//
// The two halves are scaled separately against their own ends rather than against a shared span,
// because a centre that is not the midpoint means the two halves are not the same length — scaled
// together, full left and full right would be different fractions of lock and the car would turn
// in more easily one way than the other.
export [[nodiscard]] inline double normaliseBipolar(const AxisCalibration& calibration, const double raw)
{
    const auto side = raw >= calibration.centre ? calibration.maximum : calibration.minimum;
    const auto span = side - calibration.centre;
    if (span == 0.0)
    {
        return 0.0;
    }

    const auto travel = std::clamp((raw - calibration.centre) / span, 0.0, 1.0);
    const auto shaped = applyEnds(travel, calibration.deadzone, calibration.saturation);

    return raw >= calibration.centre ? shaped : -shaped;
}

// The load cell, and the one place the difference between force and travel is stated.
//
// A potentiometer pedal reads *where the pedal is*, so the driver learns a position and threshold
// braking is a memory of geometry. A load cell reads *how hard it is being pushed*, and brake line
// pressure is proportional to pedal force through the master cylinder — so the sensor is already
// reading the quantity the hydraulics are linear in.
//
// Which is why `shape` defaults to 1. The brief this was written to asks for a non-linear curve;
// the honest default is linear, because the non-linearity a pot pedal needs is exactly the
// force-against-travel curve of the elastomer stack that a load cell has already measured past.
// The exponent stays because it is a real preference — a driver with a heavy leg wants more of the
// range spent below half pressure — and it is a preference, so it is data and it starts at the
// value that says nothing.
export struct BrakeCurve
{
    // The force at the pedal face, in newtons, that the axis's full travel corresponds to. What
    // converts a fraction of the sensor's range into the thing the sensor is measuring.
    double sensorFullScale = 900.0;
    // The force the driver means by "maximum braking". Below the cell's full scale, because the
    // usable part of a load cell is the part a leg will push through repeatably and not the part it
    // will reach once. This is the number a driver moves.
    double maximumForce = 450.0;
    // Pressure against force. 1 is what the hydraulics do.
    double shape = 1.0;
};

// Force at the pedal face, newtons, for a [0, 1] axis travel.
export [[nodiscard]] inline double pedalForce(const BrakeCurve& curve, const double travel)
{
    return std::clamp(travel, 0.0, 1.0) * curve.sensorFullScale;
}

// Brake pressure demand, [0, 1], for a [0, 1] axis travel.
export [[nodiscard]] inline double brakePressure(const BrakeCurve& curve, const double travel)
{
    if (curve.maximumForce <= 0.0)
    {
        return 0.0;
    }

    const auto fraction = std::clamp(pedalForce(curve, travel) / curve.maximumForce, 0.0, 1.0);

    return curve.shape == 1.0 ? fraction : std::pow(fraction, curve.shape);
}

// The rim's travel against the car's.
//
// `device` is what the wheel is set to lock to lock — 900 degrees on this base, and read from it
// rather than assumed. `vehicle` is what this car's own rim would have to turn for the rack to
// reach its stops.
export struct SteeringGeometry
{
    double deviceDegrees = 900.0;
    double vehicleDegrees = 900.0;
};

// A rim position in [-1, 1] of the device's own travel, to a rack demand in [-1, 1].
//
// Two cases, and they are not symmetric. A device with *more* range than the car needs maps one to
// one — the car reaches full lock partway to the stop and the rest of the rim's travel is past
// lock, which is what a real car does and what makes the wheel's degrees mean the car's degrees. A
// device with *less* range is the case where one to one costs the car lock it has: a 360 degree
// wheel on a 900 degree car could never be steered past forty percent, so the rim's stops are
// mapped to the car's stops instead. The rule is one `max`, and it is the whole of it.
export [[nodiscard]] inline double rackFromRim(const SteeringGeometry& geometry, const double rimFraction)
{
    if (geometry.vehicleDegrees <= 0.0 || geometry.deviceDegrees <= 0.0)
    {
        return std::clamp(rimFraction, -1.0, 1.0);
    }

    const auto scale = std::max(1.0, geometry.deviceDegrees / geometry.vehicleDegrees);

    return std::clamp(rimFraction * scale, -1.0, 1.0);
}

// A button, named by what it does rather than by where it is. Persisted as an index into the
// device's own button bitmap, which is the one part of a profile that is genuinely per-device.
export enum class DriverAction : std::uint8_t { Upshift, Downshift, Handbrake, Count };

export inline constexpr std::size_t driverActionCount = 3;

export [[nodiscard]] constexpr const char* driverActionName(const DriverAction action)
{
    switch (action)
    {
    case DriverAction::Upshift:
        return "upshift";
    case DriverAction::Downshift:
        return "downshift";
    case DriverAction::Handbrake:
        return "handbrake";
    case DriverAction::Count:
        break;
    }

    return "nothing";
}

// The whole of what is remembered about one device, and the whole of what is written to disk.
export struct DeviceProfile
{
    DeviceIdentity identity;
    // Free text, for whoever opens the file. Nothing is keyed on it.
    std::string name;
    std::array<AxisCalibration, inputAxisCount> axes{};
    BrakeCurve brake;
    // Lock to lock, degrees. What the device said when the profile was written, so a wheel whose
    // range was changed on the base afterwards is a profile that has to be told.
    double rotationDegrees = 900.0;
    // Newton metres at the rim for a torque fraction of one. Eight is this base's rating. It is
    // here rather than in the backend because no platform reports it and because it is the number
    // every future force is stated against — which makes it exactly the sort of thing that must
    // exist before the tuning does, not after.
    double peakTorque = 8.0;
    // Bit index into `DeviceSample::buttons`, or -1 for an action this device does not carry.
    std::array<std::int32_t, driverActionCount> buttons{{-1, -1, -1}};
};

namespace
{

[[nodiscard]] inline std::vector<std::string_view> profileTokens(const std::string_view line)
{
    auto tokens = std::vector<std::string_view>();
    auto index = std::size_t{0};

    while (index < line.size())
    {
        while (index < line.size() && (line[index] == ' ' || line[index] == '\t' || line[index] == '\r'))
        {
            index++;
        }

        if (index >= line.size() || line[index] == '#')
        {
            break;
        }

        const auto start = index;
        while (index < line.size() && line[index] != ' ' && line[index] != '\t' && line[index] != '\r')
        {
            index++;
        }

        tokens.push_back(line.substr(start, index - start));
    }

    return tokens;
}

[[nodiscard]] inline bool parseDouble(const std::string_view token, double& value)
{
    const auto result = std::from_chars(token.data(), token.data() + token.size(), value);

    return result.ec == std::errc{} && result.ptr == token.data() + token.size();
}

[[nodiscard]] inline bool parseHex16(const std::string_view token, std::uint16_t& value)
{
    auto wide = 0u;
    const auto result = std::from_chars(token.data(), token.data() + token.size(), wide, 16);
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size() || wide > 0xFFFFu)
    {
        return false;
    }

    value = static_cast<std::uint16_t>(wide);

    return true;
}

[[nodiscard]] inline bool parseIndex(const std::string_view token, std::int32_t& value)
{
    const auto result = std::from_chars(token.data(), token.data() + token.size(), value);

    return result.ec == std::errc{} && result.ptr == token.data() + token.size();
}

// std::to_string on a double is six decimals of fixed notation, which loses a raw axis end above a
// million and writes eight characters of nothing below one. to_chars' shortest round-trip form is
// what makes a written profile read back as the same numbers.
[[nodiscard]] inline std::string number(const double value)
{
    auto buffer = std::array<char, 32>{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);

    return std::string(buffer.data(), result.ptr);
}

[[nodiscard]] inline std::string hex16(const std::uint16_t value)
{
    auto buffer = std::array<char, 8>{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, 16);
    auto text = std::string(buffer.data(), result.ptr);

    return std::string(4 - std::min<std::size_t>(4, text.size()), '0') + text;
}

} // namespace

// The profile's own version. A file stating anything else is refused rather than read as though the
// fields still meant what they used to.
export inline constexpr int deviceProfileVersion = 1;

// Line-oriented, `#` to end of line, tokens separated by whitespace — the same shape as the `.cube`
// parser and for the same reason: a text format with a grammar is the thing that can be pinned by
// tests with no device anywhere near it, and every failure below is a file somebody edited by hand.
export [[nodiscard]] inline std::expected<DeviceProfile, std::string> parseDeviceProfile(const std::string_view source)
{
    auto profile = DeviceProfile{};
    auto sawVersion = false;
    auto sawIdentity = false;
    auto lineNumber = std::size_t{0};
    auto offset = std::size_t{0};

    while (offset <= source.size())
    {
        const auto end = source.find('\n', offset);
        const auto line = source.substr(offset, end == std::string_view::npos ? std::string_view::npos : end - offset);
        offset = end == std::string_view::npos ? source.size() + 1 : end + 1;
        lineNumber++;

        const auto tokens = profileTokens(line);
        if (tokens.empty())
        {
            continue;
        }

        const auto at = [&]
        {
            return " on line " + std::to_string(lineNumber);
        };
        const auto keyword = tokens[0];

        if (keyword == "version")
        {
            auto stated = 0.0;
            if (tokens.size() != 2 || !parseDouble(tokens[1], stated))
            {
                return std::unexpected("version takes one whole number" + at());
            }

            if (stated != static_cast<double>(deviceProfileVersion))
            {
                return std::unexpected("this profile states version " + number(stated) + " and this engine writes " +
                                       std::to_string(deviceProfileVersion) + "; delete it and it will be rewritten");
            }

            sawVersion = true;
            continue;
        }

        if (!sawVersion)
        {
            return std::unexpected("a profile states its version before anything else" + at());
        }

        if (keyword == "identity")
        {
            if (tokens.size() != 3 || !parseHex16(tokens[1], profile.identity.vendor) ||
                !parseHex16(tokens[2], profile.identity.product))
            {
                return std::unexpected("identity takes a vendor and a product, four hex digits each" + at());
            }

            sawIdentity = true;
            continue;
        }

        if (keyword == "name")
        {
            const auto first = line.find("name") + 4;
            const auto text = line.substr(first);
            const auto begin = text.find_first_not_of(" \t");
            const auto last = text.find_last_not_of(" \t\r");
            profile.name =
                begin == std::string_view::npos ? std::string() : std::string(text.substr(begin, last - begin + 1));
            continue;
        }

        if (keyword == "rotation" || keyword == "peak-torque")
        {
            auto value = 0.0;
            if (tokens.size() != 2 || !parseDouble(tokens[1], value))
            {
                return std::unexpected(std::string(keyword) + " takes one number" + at());
            }

            if (value <= 0.0)
            {
                return std::unexpected(std::string(keyword) + " is a magnitude and must be above zero" + at());
            }

            (keyword == "rotation" ? profile.rotationDegrees : profile.peakTorque) = value;
            continue;
        }

        if (keyword == "axis")
        {
            if (tokens.size() != 7)
            {
                return std::unexpected("axis takes a role, then minimum, maximum, centre, deadzone and saturation" +
                                       at());
            }

            const auto role = inputAxisFromName(tokens[1]);
            if (!role)
            {
                return std::unexpected("axis names a role this engine does not have: '" + std::string(tokens[1]) +
                                       "'. It takes steering, throttle, brake or clutch" + at());
            }

            auto calibration = AxisCalibration{};
            if (!parseDouble(tokens[2], calibration.minimum) || !parseDouble(tokens[3], calibration.maximum) ||
                !parseDouble(tokens[4], calibration.centre) || !parseDouble(tokens[5], calibration.deadzone) ||
                !parseDouble(tokens[6], calibration.saturation))
            {
                return std::unexpected("axis takes five numbers after the role" + at());
            }

            if (calibration.deadzone < 0.0 || calibration.saturation < 0.0 ||
                calibration.deadzone + calibration.saturation >= 1.0)
            {
                return std::unexpected("the deadzone and the saturation together leave no travel between them" + at());
            }

            profile.axes[axisIndex(*role)] = calibration;
            continue;
        }

        if (keyword == "brake-curve")
        {
            if (tokens.size() != 4 || !parseDouble(tokens[1], profile.brake.sensorFullScale) ||
                !parseDouble(tokens[2], profile.brake.maximumForce) || !parseDouble(tokens[3], profile.brake.shape))
            {
                return std::unexpected("brake-curve takes the cell's full scale, the maximum force and the shape" +
                                       at());
            }

            if (profile.brake.maximumForce <= 0.0 || profile.brake.sensorFullScale <= 0.0 || profile.brake.shape <= 0.0)
            {
                return std::unexpected("every figure on brake-curve is a magnitude and must be above zero" + at());
            }

            continue;
        }

        if (keyword == "button")
        {
            auto index = std::int32_t{-1};
            if (tokens.size() != 3 || !parseIndex(tokens[2], index))
            {
                return std::unexpected("button takes an action and a bit index" + at());
            }

            auto matched = false;
            for (auto action = std::size_t{0}; action < driverActionCount; action++)
            {
                if (tokens[1] == driverActionName(static_cast<DriverAction>(action)))
                {
                    profile.buttons[action] = index;
                    matched = true;
                }
            }

            if (!matched)
            {
                return std::unexpected("button names an action this engine does not have: '" + std::string(tokens[1]) +
                                       "'" + at());
            }

            continue;
        }

        return std::unexpected("'" + std::string(keyword) + "' is not part of a device profile" + at());
    }

    if (!sawVersion)
    {
        return std::unexpected("this is not a device profile: it states no version");
    }

    if (!sawIdentity)
    {
        return std::unexpected("this profile names no device: it states no identity");
    }

    return profile;
}

export [[nodiscard]] inline std::string deviceProfileToText(const DeviceProfile& profile)
{
    auto text = std::string("# RaceEngine input device profile.\n");
    text += "# Keyed on vendor and product, never on where the platform put the device.\n";
    text += "version " + std::to_string(deviceProfileVersion) + "\n";
    text += "identity " + hex16(profile.identity.vendor) + " " + hex16(profile.identity.product) + "\n";

    if (!profile.name.empty())
    {
        text += "name " + profile.name + "\n";
    }

    text += "rotation " + number(profile.rotationDegrees) + "\n";
    text += "peak-torque " + number(profile.peakTorque) + "\n";

    for (auto index = std::size_t{0}; index < inputAxisCount; index++)
    {
        const auto& axis = profile.axes[index];
        text += "axis ";
        text += inputAxisName(static_cast<InputAxis>(index));
        text += " " + number(axis.minimum) + " " + number(axis.maximum) + " " + number(axis.centre) + " " +
                number(axis.deadzone) + " " + number(axis.saturation) + "\n";
    }

    text += "brake-curve " + number(profile.brake.sensorFullScale) + " " + number(profile.brake.maximumForce) + " " +
            number(profile.brake.shape) + "\n";

    // Seeded unbound rather than guessed. A device's buttons are the one part of it that says
    // nothing about itself — this base reports a hundred and eight of them and names none — so a
    // default would be a wheel that shifts when some unrelated button is pressed, which is worse to
    // diagnose than one that does not shift at all.
    text += "# -1 is unbound. To find a button's index, run the engine's test binary as\n";
    text += "#   EngineTests \"[device]\"\n";
    text += "# holding the one you want: it prints the bitmap it saw, and the index is the bit.\n";

    for (auto index = std::size_t{0}; index < driverActionCount; index++)
    {
        text += "button ";
        text += driverActionName(static_cast<DriverAction>(index));
        text += " " + std::to_string(profile.buttons[index]) + "\n";
    }

    return text;
}

// The file this device's profile is kept in. Identity and nothing else: the same wheel on the same
// desk keeps its calibration across a replug, across a reboot that renumbered the event nodes, and
// across the two platforms.
export [[nodiscard]] inline std::string deviceProfileFileName(const DeviceIdentity identity)
{
    return hex16(identity.vendor) + "-" + hex16(identity.product) + ".profile";
}

// What a device gets before anybody has calibrated it: the ends the device itself states, the
// resting sample as the centre of the rim and as the released end of every pedal.
//
// Taking rest as "released" rather than assuming a direction is the one rule here that is not
// arbitrary. This base reads 65535 on all three pedal axes with nothing on them, and a base with no
// pedals attached reads exactly the same — so no fixed polarity is right for both, and the only
// thing that is true in either case is that whatever the axis reads when the device is opened is
// what it reads when nothing is being asked of it.
export [[nodiscard]] inline DeviceProfile seedDeviceProfile(const DeviceDescription& description,
                                                            const DeviceSample& rest)
{
    auto profile = DeviceProfile{};
    profile.identity = description.identity;
    profile.name = description.name;

    if (description.rotationDegrees > 0.0)
    {
        profile.rotationDegrees = description.rotationDegrees;
    }

    for (auto index = std::size_t{0}; index < inputAxisCount; index++)
    {
        const auto& bounds = description.axes[index];
        if (!bounds.present)
        {
            continue;
        }

        const auto low = static_cast<double>(bounds.minimum);
        const auto high = static_cast<double>(bounds.maximum);
        const auto resting = static_cast<double>(rest.axes[index]);
        auto& calibration = profile.axes[index];

        if (static_cast<InputAxis>(index) == InputAxis::Steering)
        {
            calibration.minimum = low;
            calibration.maximum = high;
            calibration.centre = resting;
            continue;
        }

        // Whichever end the axis is resting nearer is the released end; the other is full travel.
        const auto restingNearLow = std::abs(resting - low) <= std::abs(resting - high);
        calibration.minimum = restingNearLow ? low : high;
        calibration.maximum = restingNearLow ? high : low;
        calibration.centre = resting;
    }

    return profile;
}

} // namespace raceengine
