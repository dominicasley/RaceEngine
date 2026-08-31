module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

module raceengine.assists;

namespace raceengine
{

[[nodiscard]] bool assistsEngaged(const AssistSetup& setup)
{
    return setup.antilock.enabled || setup.traction.mode != TractionMode::Off || setup.cornering.enabled;
}

namespace
{

[[nodiscard]] BrakeChannel channelAt(const std::size_t index)
{
    switch (index)
    {
    case 0:
        return BrakeChannel::FrontLeft;
    case 1:
        return BrakeChannel::FrontRight;
    default:
        break;
    }

    return BrakeChannel::Rear;
}

} // namespace

[[nodiscard]] AssistOutput updateAssists(const AssistSetup& setup, AssistState& state, const AssistSensors& sensors,
                                         const AssistDemand& demand,
                                         const std::array<double, wheelCount>& wheelPressure, const double deltaTime)
{
    auto output = AssistOutput{};

    const auto braking = demand.brake > setup.brakeSwitch;

    // What the pedal has put at each caliper, and what each corner's brake is worth if the pedal goes
    // all the way. The second is what the two torque-based controllers below are calibrated in;
    // derived here rather than stored, so there is one statement of this car's brakes.
    auto driverPressure = std::array<double, wheelCount>{};
    auto peakTorque = std::array<double, wheelCount>{};

    for (auto wheel = std::size_t{0}; wheel < wheelCount; wheel++)
    {
        driverPressure[wheel] = std::max(wheelPressure[wheel], 0.0);
        peakTorque[wheel] = setup.brakeTorquePerPressure[wheel] * std::max(setup.maximumWheelPressure[wheel], 0.0);
    }

    // --- the controller's own clock -----------------------------------------------------------
    //
    // Whole periods only, with the remainder carried. A physics tick shorter than one period runs
    // nothing and the actuator holds, which is what a solenoid does between commands.
    const auto period = setup.controlRate > 0.0 ? 1.0 / setup.controlRate : deltaTime;

    state.clockRemainder += deltaTime;

    // What everything upstream wanted at each wheel on the last controller step, kept out here so
    // the telemetry below can report what the modulator did to it. Seeded with the pedal so a tick
    // too short to run any step at all still reports against something true.
    auto requested = driverPressure;

    // **The tone rings are sampled on the controller's clock, not on the physics tick's**, and that
    // is not tidiness. Sampled once per 2.78 ms tick and then read by three 1 ms controller steps,
    // the reference speed sees two steps where nothing moved and one where it dropped 25 m/s^2 — and
    // that third step is above the estimator's own fall limit, so it is clamped, marked as
    // limiter-carried and excluded from the rate it learns. The other two teach it zero. Measured:
    // the ECU believed the car was decelerating at 0.39 m/s^2 through a 0.9 g stop, every threshold
    // measured against that fired, and a dry 100-0 came out at 85 m against 41 with 119 spurious
    // cycles on wheels that physically cannot lock.
    //
    // A real ECU reads its capture registers on its own timer and the pulses land between reads,
    // which is exactly this. The wheel's rotation is taken as uniform across the physics tick — the
    // vehicle integrates it once per tick — so splitting that tick's angle across the controller's
    // steps loses nothing.
    auto readings = WheelSpeedReadings{};
    auto stepped = false;

    while (state.clockRemainder >= period)
    {
        state.clockRemainder -= period;
        stepped = true;

        readings = sampleWheelSensors(setup.toneRing, state.sensors, sensors.wheelSpeeds, period);

        advanceReferenceSpeed(setup.reference, state.reference, readings, braking, period);

        advanceTractionControl(setup.traction, state.traction, readings, setup.reference, peakTorque,
                               state.reference.speed, state.reference.valid, state.reference.coasting, demand.throttle,
                               period);

        advanceCorneringBrake(setup.cornering, state.cornering, readings, setup.reference, peakTorque,
                              sensors.lateralAcceleration, period);

        // What everything upstream wants at each wheel, before the modulator has its say. The
        // driver's foot and the two brake-based interventions arrive at the same caliper, so they
        // combine into one requested pressure rather than into three commands — which is also the
        // seam a stability controller drops into without touching anything here.
        for (auto wheel = std::size_t{0}; wheel < wheelCount; wheel++)
        {
            const auto intervention =
                std::min(1.0, state.traction.brakeFraction[wheel] + state.cornering.brakeFraction[wheel]) *
                std::max(setup.maximumWheelPressure[wheel], 0.0);

            requested[wheel] = std::max(driverPressure[wheel], intervention);
        }

        // The yaw moment build-up delay, stepped before the channels because it is the one part of
        // this controller that reads across them: the high front wheel's request is capped by the
        // low front wheel's signals. Reading the phases as they stand — last period's — is what
        // gives it the one-period lag a real unit's captured samples have.
        //
        // **Off on every car**, so `ceiling` is 0.0, `engaged` is false and the cap below is never
        // taken.
        auto frontRequests = std::array<double, 2>{};
        for (auto index = std::size_t{0}; index < 2; index++)
        {
            const auto channel = channelAt(index);
            for (auto wheel = std::size_t{0}; wheel < wheelCount; wheel++)
            {
                if (antilockDrivesWheel(channel, wheel))
                {
                    frontRequests[index] = std::max(frontRequests[index], requested[wheel]);
                }
            }
        }

        const auto ceiling =
            advanceYawMomentDelay(setup.antilock, state.antilock.yawDelay, state.antilock.channels[0],
                                  state.antilock.channels[1], frontRequests, sensors.lateralAcceleration, braking,
                                  period);

        for (auto index = std::size_t{0}; index < brakeChannelCount; index++)
        {
            const auto channel = channelAt(index);
            const auto controlWheel = antilockControlWheel(channel, readings);

            // The channel serves whichever wheels it drives, so it must answer the highest request
            // among them; select-low is about which wheel is *measured*, not about which request is
            // honoured.
            auto channelRequest = 0.0;
            for (auto wheel = std::size_t{0}; wheel < wheelCount; wheel++)
            {
                if (antilockDrivesWheel(channel, wheel))
                {
                    channelRequest = std::max(channelRequest, requested[wheel]);
                }
            }

            // And the delay's staged ceiling, on the high front channel alone. Every other channel
            // — the low front and the rear — is untouched by it, which is the whole mechanism: the
            // low wheel is already being modulated by its own signals and the rear axle has no part
            // in a left-to-right asymmetry.
            if (state.antilock.yawDelay.engaged && index == state.antilock.yawDelay.highChannel)
            {
                channelRequest = std::min(channelRequest, ceiling);
            }

            const auto wheelSpeed = std::abs(sensedRoadSpeed(setup.reference, readings[controlWheel]));

            const auto pressure = advanceAntilockChannel(
                setup.antilock, channel, state.antilock.channels[index], readings[controlWheel], wheelSpeed,
                state.reference.speed, state.reference.rate, state.reference.valid, channelRequest, period);

            for (auto wheel = std::size_t{0}; wheel < wheelCount; wheel++)
            {
                if (antilockDrivesWheel(channel, wheel))
                {
                    state.pressure[wheel] = pressure;
                }
            }
        }
    }

    // A tick too short to contain one controller period runs nothing and advances nothing; the
    // actuator holds and the telemetry reports the reading the ECU is still sitting on. Sampling
    // with a zero step reads it back without moving the sensor on.
    if (!stepped)
    {
        readings = sampleWheelSensors(setup.toneRing, state.sensors, sensors.wheelSpeeds, 0.0);
    }

    // --- what the actuators are at -------------------------------------------------------------
    output.brakes.commanded = assistsEngaged(setup);
    output.throttleScale = tractionThrottleScale(state.traction);
    output.channels.yawDelayEngaged = state.antilock.yawDelay.engaged;
    output.channels.yawDelayCeiling = state.antilock.yawDelay.engaged ? state.antilock.yawDelay.ceiling : 0.0;

    for (auto wheel = std::size_t{0}; wheel < wheelCount; wheel++)
    {
        const auto perPressure = setup.brakeTorquePerPressure[wheel];
        const auto peak = peakTorque[wheel];

        output.brakes.wheels[wheel] = state.pressure[wheel] * perPressure;

        auto& channels = output.channels;
        channels.pressure[wheel] = state.pressure[wheel];
        channels.sensedWheelSpeed[wheel] = sensedRoadSpeed(setup.reference, readings[wheel]);
        channels.sensorAge[wheel] = readings[wheel].age;
        channels.estimatedSlip[wheel] =
            estimatedSlip(state.reference.speed, std::abs(channels.sensedWheelSpeed[wheel]));

        // Reported by source. The driver's is what the pedal alone would have made; the anti-lock
        // figure is what the modulator took off what it was asked for, and is negative-going by
        // construction because that unit can only ever remove pressure.
        channels.driverBrakeTorque[wheel] = driverPressure[wheel] * perPressure;
        channels.antilockBrakeTorque[wheel] = (state.pressure[wheel] - requested[wheel]) * perPressure;
        channels.tractionBrakeTorque[wheel] = state.traction.brakeFraction[wheel] * peak;
        channels.corneringBrakeTorque[wheel] = state.cornering.brakeFraction[wheel] * peak;
    }

    for (auto index = std::size_t{0}; index < brakeChannelCount; index++)
    {
        const auto channel = channelAt(index);
        const auto& channelState = state.antilock.channels[index];

        for (auto wheel = std::size_t{0}; wheel < wheelCount; wheel++)
        {
            if (!antilockDrivesWheel(channel, wheel))
            {
                continue;
            }

            output.channels.antilockActive[wheel] = channelState.phase != ModulatorPhase::Passive;
            output.channels.antilockCycles[wheel] = channelState.cycles;
            output.channels.antilockPhase[wheel] = channelState.phase;
            output.channels.sensedWheelAcceleration[wheel] = channelState.acceleration;
        }
    }

    output.channels.referenceSpeed = state.reference.speed;
    output.channels.referenceValid = state.reference.valid;
    output.channels.referenceCoasting = state.reference.coasting;
    output.channels.referenceAcceleration = state.reference.rate;
    output.channels.engineTorqueReduction = state.traction.engineReduction;
    output.channels.tractionBrakeActive = state.traction.brakeActive;
    output.channels.tractionEngineActive = state.traction.engineActive;
    output.channels.corneringActive = state.cornering.active;

    return output;
}

} // namespace raceengine
