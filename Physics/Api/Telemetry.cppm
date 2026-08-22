module;

#include <array>
#include <charconv>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.physics:Telemetry;

namespace raceengine
{

// What the car was doing, tick by tick.
//
// This is not instrumentation bolted onto a finished model — it is how every acceptance criterion
// after the integrator is *checked*. A skidpad that "feels about right" is not a result; a plot of
// lateral acceleration against steering angle with a monotonic gradient and a defined limit is. So
// the channels are chosen to answer the questions the milestone asks, and the ring is written every
// tick whether anything is looking or not.
//
// SI in, SI out, with one exception noted at the CSV boundary: the columns a human reads are in the
// units that human expects — degrees, km/h, g — because a channel nobody can read is a channel
// nobody checks.

export enum class Corner : std::uint32_t { FrontLeft, FrontRight, RearLeft, RearRight };

export inline constexpr std::size_t cornerCount = 4;

export [[nodiscard]] constexpr const char* cornerAbbreviation(const Corner corner)
{
    switch (corner)
    {
    case Corner::FrontLeft:
        return "FL";
    case Corner::FrontRight:
        return "FR";
    case Corner::RearLeft:
        return "RL";
    case Corner::RearRight:
        return "RR";
    }

    return "??";
}

export struct WheelTelemetry
{
    // The channel every other one is read against: if the load is wrong, the tire's answer is wrong
    // however good the tire model is.
    double verticalLoad = 0.0;

    double slipRatio = 0.0;
    double slipAngle = 0.0;

    double forceLongitudinal = 0.0;
    double forceLateral = 0.0;
    double aligningMoment = 0.0;

    double suspensionTravel = 0.0;
    double damperVelocity = 0.0;
    double angularVelocity = 0.0;

    // Not in the brief's minimum, and here anyway: camber and grip are what a load-sensitivity or
    // surface-blending fault shows up in first, and both were expensive to add later in every
    // engine that did.
    double camber = 0.0;
    double gripMultiplier = 1.0;

    bool inContact = false;

    // How many of the contact patch's grid samples are actually touching, out of the grid the tyre
    // is sampled on. `inContact` is this being non-zero; the count is the channel, because the two
    // answer different questions and only the count answers the one that is open.
    //
    // **It is here to size a milestone rather than to plot a force.** The kerb-edge load dip is the
    // spring bed collapsing where samples fall off a chamfer with the missed ones left in the
    // divisor, and every measurement of it so far was taken on a synthetic probe against an analytic
    // edge. Whether real driving enters that regime at all is a question about laps, not about
    // probes, and this is the one integer that answers it.
    std::uint32_t contactingSamples = 0;

    // Deepest minus shallowest of those contacting samples, in metres. The count says *how much* of
    // the patch is carrying; this says whether what it is carrying is **tilted**, and the pair is
    // what separates a wheel loaded across a kerb chamfer from a wheel resting lightly on flat road.
    //
    // Both of those report a low `verticalLoad` today and only one of them should, so the count on
    // its own selects the wrong ticks. `verticalLoad` cannot arbitrate because it is derived from
    // the aggregate that the kerb case corrupts — the full argument is on
    // `ContactPatch::depthSpread`, which is where this is taken from.
    double patchDepthSpread = 0.0;
};

export struct TelemetryFrame
{
    double time = 0.0;
    std::uint64_t tick = 0;

    // World frame, both of them: where the car is and how fast it is going over the ground.
    glm::dvec3 position{0.0};
    glm::dvec3 velocity{0.0};

    // **The car's own frame**, and it was the world's until 2026-08-21. The CSV writes `.z` into
    // `G Force Long` and `.x` into `G Force Lat`, which are body-frame roles — correct on a straight
    // aimed along world +z, which is every fixture that has ever looked at them, and wrong almost
    // everywhere on a circuit. Driven a quarter turn apart the same manoeuvre reported its
    // longitudinal deceleration in the lateral column and back again: −6.44/−4.01 became
    // +4.01/−6.44. Rotated here, at the one place that knows the attitude.
    glm::dvec3 acceleration{0.0};

    // Yaw, pitch and roll, and their rates. Held as Euler angles rather than as the chassis
    // quaternion because this is the channel a human reads and argues about; the quaternion is in
    // the state, where it belongs.
    double yaw = 0.0;
    double pitch = 0.0;
    double roll = 0.0;
    double yawRate = 0.0;
    double pitchRate = 0.0;
    double rollRate = 0.0;

    // The axle means off `VehicleStep::rideHeight`, in metres. **Computed since M1 and reaching no
    // channel until now** — this file's own map claimed it was "computed and passed" and the second
    // half of that was not true, which is exactly the kind of claim a column makes checkable. Zero
    // with no corner touching, because there is then no road to measure from.
    double rideHeightFront = 0.0;
    double rideHeightRear = 0.0;

    // The driver's demand, −1 to 1, dimensionless — and it stays that because that is what it is.
    double steering = 0.0;
    // What the steering *wheel* is at, radians: the demand through this car's own lock-to-lock.
    // Separate from the demand rather than replacing it, because the two answer different questions
    // and conflating them is how the column below came to carry a demand in a field named degrees.
    double steeringWheelAngle = 0.0;
    double throttle = 0.0;
    double brake = 0.0;
    // Where the clutch pedal actually ended up, 0 engaged to 1 fully depressed — which is the
    // auto-clutch's answer and not the driver's demand whenever nobody is on the pedal. Filled
    // beside the driveline's other channels, from the state that owns it.
    double clutch = 0.0;
    std::int32_t gear = 0;
    // Where the gearbox is between two gears, as `ShiftPhase`'s own order: 0 engaged, 1 disengaging,
    // 2 neutral, 3 engaging. A number rather than the enum because this partition names nothing it
    // does not import and `:Driveline` imports *this* one, not the other way about — and a channel
    // is a number by the time anybody plots it anyway.
    std::uint32_t shiftPhase = 0;
    double engineSpeed = 0.0;

    // The driveline's channels. Filled by whoever stepped the driveline: the vehicle tick cannot,
    // because `:Vehicle` does not import `:Driveline` and must not.
    double engineTorque = 0.0;
    double clutchTorque = 0.0;
    double clutchSlip = 0.0;
    // Accumulated rather than per tick, and in joules. It is the hook a clutch thermal model reads
    // and nothing consumes it yet; a channel that had to be integrated by whoever plotted it is a
    // channel that gets integrated differently twice.
    double clutchSlipEnergy = 0.0;

    std::array<WheelTelemetry, cornerCount> wheels{};
};

static_assert(std::is_trivially_copyable_v<TelemetryFrame>,
              "frames are copied into a ring every tick and must cost nothing to copy");

// A ring, because a validation run is minutes long at 360 Hz and the interesting part is usually
// the end. Fixed capacity taken at construction: a recorder that grew would allocate inside the
// tick, which is the one place nothing may.
export class TelemetryRecorder
{
public:
    explicit TelemetryRecorder(const std::size_t capacity) :
        frames(capacity)
    {
    }

    void record(const TelemetryFrame& frame)
    {
        if (frames.empty())
        {
            return;
        }

        frames[next] = frame;
        next = (next + 1) % frames.size();
        filled = filled < frames.size() ? filled + 1 : frames.size();
    }

    // Oldest first, which is the order anything reading this wants and the opposite of the order
    // the ring stores them in once it has wrapped.
    [[nodiscard]] std::vector<TelemetryFrame> inOrder() const
    {
        auto ordered = std::vector<TelemetryFrame>{};
        ordered.reserve(filled);

        const auto start = filled < frames.size() ? std::size_t{0} : next;
        for (auto offset = std::size_t{0}; offset < filled; offset++)
        {
            ordered.push_back(frames[(start + offset) % frames.size()]);
        }

        return ordered;
    }

    [[nodiscard]] std::size_t size() const
    {
        return filled;
    }

    [[nodiscard]] std::size_t capacity() const
    {
        return frames.size();
    }

    void clear()
    {
        next = 0;
        filled = 0;
    }

private:
    std::vector<TelemetryFrame> frames;
    std::size_t next = 0;
    std::size_t filled = 0;
};

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

export [[nodiscard]] std::string telemetryToCsv(const std::vector<TelemetryFrame>& frames)
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
            "Engine Torque [Nm],Clutch Torque [Nm],Clutch Slip [rad/s],Clutch Slip Energy [J]";

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
        }

        text += "\n";
    }

    return text;
}

} // namespace raceengine
