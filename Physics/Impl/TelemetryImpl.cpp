// Telemetry bodies. Declarations are in Api/Telemetry.cppm.
//
// A **module implementation unit** — `module raceengine.physics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface partition is part of the module's BMI instead, and editing one rebuilt every importer
// of `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <array>
#include <charconv>
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <type_traits>
#include <vector>

module raceengine.physics;

namespace raceengine
{

// CSV, and a *pure* function of the frames rather than something that writes a file: what is in the
// text is worth testing and opening a file is not. The caller writes it.
//
// Column names follow MoTeC i2's conventions where they exist — "Susp Pos FL", "Damper Vel FL",
// "G Force Lat", "Engine RPM" — so a run can be dropped into the tooling a race engineer already
// has. Where MoTeC has no name for something this model exposes, the name is spelled out rather
// than abbreviated into a guess.
namespace
{

// Locale independent, and that is the whole reason for reaching past the obvious stream: a CSV
// written on a machine with a comma decimal separator is not a CSV, and the failure is invisible
// until someone else opens the file.
void appendNumber(std::string& text, const double value, const int precision = 6)
{
    auto buffer = std::array<char, 64>{};
    const auto written =
        std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::fixed, precision);

    text.append(buffer.data(), written.ptr);
}

void appendInteger(std::string& text, const long long value)
{
    auto buffer = std::array<char, 32>{};
    const auto written = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);

    text.append(buffer.data(), written.ptr);
}

constexpr auto radiansToDegrees = 57.29577951308232;
constexpr auto metresPerSecondToKilometresPerHour = 3.6;
// The column has always said rpm and used to carry rad/s, which is a factor of 9.55 in a channel
// every engine number is read against.
constexpr auto radiansPerSecondToRevolutionsPerMinute = 9.549296585513721;
constexpr auto gravity = 9.80665;

} // namespace

[[nodiscard]] std::string telemetryToCsv(const std::vector<TelemetryFrame>& frames)
{
    auto text = std::string{};
    text.reserve(frames.size() * 512 + 2048);

    // The units are in the header because a channel whose units are a matter of memory is a channel
    // that gets misread. Angles in degrees, speed in km/h, accelerations in g and travel in
    // millimetres: not the SI the model works in, but the units the person reading the plot thinks
    // in, and the conversion happens once here rather than in everybody's head.
    text += "Time [s],Tick,"
            "Pos X [m],Pos Y [m],Pos Z [m],"
            "Speed [km/h],Vel X [m/s],Vel Y [m/s],Vel Z [m/s],"
            "G Force Long [g],G Force Lat [g],G Force Vert [g],"
            "Yaw [deg],Pitch [deg],Roll [deg],"
            "Yaw Rate [deg/s],Pitch Rate [deg/s],Roll Rate [deg/s],"
            "Ride Height F [mm],Ride Height R [mm],"
            "Steering Angle [deg],Steering Demand [],Throttle Pos [%],Brake Pos [%],Clutch Pos [%],Gear,"
            "Shift Phase [],"
            "Engine RPM [rpm],"
            "Engine Torque [Nm],Clutch Torque [Nm],Clutch Slip [rad/s],Clutch Slip Energy [J],"
            "Reference Speed [km/h],Engine Cut [%]";

    for (auto corner = std::size_t{0}; corner < cornerCount; corner++)
    {
        const auto tag = std::string(cornerAbbreviation(static_cast<Corner>(corner)));

        text += ",Tyre Load " + tag + " [N]";
        text += ",Slip Ratio " + tag + " []";
        text += ",Slip Angle " + tag + " [deg]";
        text += ",Tyre Force X " + tag + " [N]";
        text += ",Tyre Force Y " + tag + " [N]";
        text += ",Aligning Moment " + tag + " [Nm]";
        text += ",Susp Pos " + tag + " [mm]";
        text += ",Damper Vel " + tag + " [mm/s]";
        text += ",Wheel Speed " + tag + " [rad/s]";
        text += ",Camber " + tag + " [deg]";
        text += ",Grip " + tag + " []";
        text += ",In Contact " + tag + " []";
        text += ",Contact Samples " + tag + " []";
        text += ",Patch Depth Spread " + tag + " [mm]";
        text += ",Brake Pressure " + tag + " []";
        text += ",ABS Torque " + tag + " [Nm]";
        text += ",TC Torque " + tag + " [Nm]";
        text += ",XDS Torque " + tag + " [Nm]";
        text += ",Sensed Wheel Speed " + tag + " [m/s]";
        text += ",Tyre Temp Surface " + tag + " [C]";
        text += ",Tyre Temp Core " + tag + " [C]";
        text += ",Tyre Temp Carcass " + tag + " [C]";
        text += ",Disc Temp " + tag + " [C]";
        text += ",Wheel Temp " + tag + " [C]";
    }

    text += "\n";

    for (const auto& frame : frames)
    {
        appendNumber(text, frame.time, 6);
        text += ",";
        appendInteger(text, static_cast<long long>(frame.tick));

        for (const auto value : {frame.position.x, frame.position.y, frame.position.z})
        {
            text += ",";
            appendNumber(text, value, 5);
        }

        text += ",";
        appendNumber(text, glm::length(frame.velocity) * metresPerSecondToKilometresPerHour, 4);
        for (const auto value : {frame.velocity.x, frame.velocity.y, frame.velocity.z})
        {
            text += ",";
            appendNumber(text, value, 5);
        }

        for (const auto value : {frame.acceleration.z, frame.acceleration.x, frame.acceleration.y})
        {
            text += ",";
            appendNumber(text, value / gravity, 5);
        }

        for (const auto value : {frame.yaw, frame.pitch, frame.roll, frame.yawRate, frame.pitchRate, frame.rollRate})
        {
            text += ",";
            appendNumber(text, value * radiansToDegrees, 4);
        }

        for (const auto value : {frame.rideHeightFront, frame.rideHeightRear})
        {
            text += ",";
            appendNumber(text, value * 1000.0, 3);
        }

        // The rim's angle, and the demand beside it. Not the same number and no longer the same
        // column: this one used to be `frame.steering * radiansToDegrees`, which is a dimensionless
        // demand times a radians conversion, and printed 28.6 for a half-lock input on a car whose
        // rim is at 189 degrees there.
        text += ",";
        appendNumber(text, frame.steeringWheelAngle * radiansToDegrees, 4);
        text += ",";
        appendNumber(text, frame.steering, 5);
        text += ",";
        appendNumber(text, frame.throttle * 100.0, 3);
        text += ",";
        appendNumber(text, frame.brake * 100.0, 3);
        text += ",";
        appendNumber(text, frame.clutch * 100.0, 3);
        text += ",";
        appendInteger(text, frame.gear);
        text += ",";
        appendInteger(text, static_cast<long long>(frame.shiftPhase));
        text += ",";
        appendNumber(text, frame.engineSpeed * radiansPerSecondToRevolutionsPerMinute, 1);
        text += ",";
        appendNumber(text, frame.engineTorque, 2);
        text += ",";
        appendNumber(text, frame.clutchTorque, 2);
        text += ",";
        appendNumber(text, frame.clutchSlip, 4);
        text += ",";
        appendNumber(text, frame.clutchSlipEnergy, 1);
        text += ",";
        appendNumber(text, frame.referenceSpeed * metresPerSecondToKilometresPerHour, 4);
        text += ",";
        appendNumber(text, frame.engineTorqueReduction * 100.0, 3);

        for (const auto& wheel : frame.wheels)
        {
            text += ",";
            appendNumber(text, wheel.verticalLoad, 2);
            text += ",";
            appendNumber(text, wheel.slipRatio, 5);
            text += ",";
            appendNumber(text, wheel.slipAngle * radiansToDegrees, 4);
            text += ",";
            appendNumber(text, wheel.forceLongitudinal, 2);
            text += ",";
            appendNumber(text, wheel.forceLateral, 2);
            text += ",";
            appendNumber(text, wheel.aligningMoment, 3);
            text += ",";
            appendNumber(text, wheel.suspensionTravel * 1000.0, 3);
            text += ",";
            appendNumber(text, wheel.damperVelocity * 1000.0, 3);
            text += ",";
            appendNumber(text, wheel.angularVelocity, 4);
            text += ",";
            appendNumber(text, wheel.camber * radiansToDegrees, 4);
            text += ",";
            appendNumber(text, wheel.gripMultiplier, 4);
            text += ",";
            appendInteger(text, wheel.inContact ? 1 : 0);
            text += ",";
            appendInteger(text, static_cast<long long>(wheel.contactingSamples));
            text += ",";
            appendNumber(text, wheel.patchDepthSpread * 1000.0, 3);
            text += ",";
            appendNumber(text, wheel.brakePressure, 4);
            text += ",";
            appendNumber(text, wheel.antilockBrakeTorque, 2);
            text += ",";
            appendNumber(text, wheel.tractionBrakeTorque, 2);
            text += ",";
            appendNumber(text, wheel.corneringBrakeTorque, 2);
            text += ",";
            appendNumber(text, wheel.sensedWheelSpeed, 4);
            text += ",";
            appendNumber(text, wheel.tyreSurfaceTemperature, 2);
            text += ",";
            appendNumber(text, wheel.tyreCoreTemperature, 2);
            text += ",";
            appendNumber(text, wheel.tyreCarcassTemperature, 2);
            text += ",";
            appendNumber(text, wheel.discTemperature, 2);
            text += ",";
            appendNumber(text, wheel.wheelTemperature, 2);
        }

        text += "\n";
    }

    return text;
}

void fillAssistTelemetry(TelemetryFrame& frame, const AssistChannels& channels)
{
    frame.referenceSpeed = channels.referenceSpeed;
    frame.engineTorqueReduction = channels.engineTorqueReduction;

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        auto& wheel = frame.wheels[index];

        wheel.brakePressure = channels.pressure[index];
        wheel.antilockBrakeTorque = channels.antilockBrakeTorque[index];
        wheel.tractionBrakeTorque = channels.tractionBrakeTorque[index];
        wheel.corneringBrakeTorque = channels.corneringBrakeTorque[index];
        wheel.sensedWheelSpeed = channels.sensedWheelSpeed[index];
    }
}

} // namespace raceengine
