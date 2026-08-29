module;

#include <array>
#include <charconv>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.physics:Telemetry;

// The assist layer's channels. Importing it here is safe and importing this from *there* is not:
// `raceengine.assists` is a module of its own precisely so that the dependency runs one way, and a
// controller that could name a `TelemetryFrame` could reach vehicle state through it.
import raceengine.assists;

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

// Which axle a corner is on. Stated once rather than as `index < 2` at each site, because the brake
// hydraulics ask the question per corner and an inverted comparison there is a car that proportions
// its front circuit.
export [[nodiscard]] constexpr bool rearAxle(const Corner corner)
{
    return corner == Corner::RearLeft || corner == Corner::RearRight;
}

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

    // The bushes' fore-aft deflection of this wheel, metres, positive forward — the state
    // `CornerSetup::longitudinalForceRecession` drives. Exactly 0.0 on a car stating no
    // coefficient, which is what makes a trace able to answer "was it on": camber's compliance
    // shows in the camber channel, and this is the only channel recession shows in at all.
    double complianceRecession = 0.0;
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

    // --- what the electronics did to this wheel ---
    //
    // **Every source on its own channel**, because "the brakes came on" is not a finding. Zero
    // throughout on a car with nothing switched on, which is every fixture but the assist suite's.
    //
    // The pressure the hydraulic unit ended the tick at, as a fraction of full system pressure, and
    // then what each system asked of the caliper in newton metres. The anti-lock figure is what the
    // modulator *removed* and is therefore zero or negative; the other two are additive.
    double brakePressure = 0.0;
    double antilockBrakeTorque = 0.0;
    double tractionBrakeTorque = 0.0;
    double corneringBrakeTorque = 0.0;

    // What the wheel speed sensor reported, through the ECU's nominal radius, in m/s. Plotted
    // against `Speed` it is the estimator's error made visible, which is the channel to read first
    // when the electronics do something inexplicable.
    double sensedWheelSpeed = 0.0;

    // The tread's three temperatures, degrees Celsius. All three, because the whole finding this
    // model rests on is that they are **not** the same number and that grip follows the middle one:
    // the surface moves within a corner, the core within a lap, the carcass within a stint. A single
    // "tyre temperature" column would hide exactly the thing worth plotting.
    //
    // They read the seed temperature on a car with `tyreThermal` off, which is what a channel should
    // say about a quantity that is not being simulated: the value the rest of the model is assuming.
    double tyreSurfaceTemperature = 0.0;
    double tyreCoreTemperature = 0.0;
    double tyreCarcassTemperature = 0.0;

    // The cavity air's temperature and the pressure that follows from it, degrees Celsius and **psi
    // gauge**. Stage 2's two channels since 2026-08-28.
    //
    // **The pressure is in psi and not pascals**, alone among this model's pressures, because it is
    // the one a driver reads off a gauge and sets by hand. A trace column exists to be looked at.
    //
    // They read the seed and the pressure it implies on a car with `tyrePressure` off, which is the
    // same contract the three temperatures above keep: the value the rest of the model is assuming.
    double tyreGasTemperature = 0.0;
    double tyrePressurePsi = 0.0;

    // The brake disc's temperature, degrees Celsius. One channel and not three: a disc is one lump of
    // iron in this model, and what a plot of it answers is "did the brakes fade", which is a question
    // about one number.
    double discTemperature = 0.0;

    // The wheel's, the same way — the node between the disc and the tyre since stage 3. It is worth
    // its own column rather than being inferred, because the whole question stage 3 answers is how
    // much of the disc's heat gets past it, and the answer is a temperature between two others.
    double wheelTemperature = 0.0;
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

    // The assist layer's two whole-car channels: what the ECU believes the road speed is, and how
    // much of the driver's throttle traction control is holding back. Filled by whoever ran the
    // assist layer, for the reason the driveline's channels are.
    double referenceSpeed = 0.0;
    double engineTorqueReduction = 0.0;

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

// The assist layer's channels into a frame the vehicle tick already filled. Separate from the tick
// for `fillDrivelineTelemetry`'s reason: the vehicle model does not know an assist layer exists, and
// whoever ran one is who can say what it did.
export void fillAssistTelemetry(TelemetryFrame& frame, const AssistChannels& channels);

export [[nodiscard]] std::string telemetryToCsv(const std::vector<TelemetryFrame>& frames);

} // namespace raceengine
