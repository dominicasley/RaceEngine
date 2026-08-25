module;

#include <algorithm>
#include <cmath>
#include <cstddef>

module raceengine.assists;

namespace raceengine
{

[[nodiscard]] std::size_t antilockControlWheel(const BrakeChannel channel, const WheelSpeedReadings& wheels)
{
    switch (channel)
    {
    case BrakeChannel::FrontLeft:
        return frontLeft;
    case BrakeChannel::FrontRight:
        return frontRight;
    case BrakeChannel::Rear:
        break;
    }

    // Select-low. A wheel with no measurement at all cannot be the one selected — it is not reading
    // slow, it is not reading — so an invalid reading loses to a valid one whatever the numbers say.
    const auto& left = wheels[rearLeft];
    const auto& right = wheels[rearRight];

    if (left.valid != right.valid)
    {
        return left.valid ? rearLeft : rearRight;
    }

    return std::abs(left.speed) <= std::abs(right.speed) ? rearLeft : rearRight;
}

[[nodiscard]] bool antilockDrivesWheel(const BrakeChannel channel, const std::size_t wheel)
{
    switch (channel)
    {
    case BrakeChannel::FrontLeft:
        return wheel == frontLeft;
    case BrakeChannel::FrontRight:
        return wheel == frontRight;
    case BrakeChannel::Rear:
        return wheel == rearLeft || wheel == rearRight;
    }

    return false;
}

[[nodiscard]] double advanceAntilockChannel(const AntilockSetup& setup, AntilockChannelState& state,
                                            const WheelSpeedReading& wheel, const double wheelRoadSpeed,
                                            const double referenceSpeed, const double referenceAcceleration,
                                            const bool referenceValid, const double requestedPressure,
                                            const double deltaTime)
{
    // --- what the sensor has told the controller since it last looked -------------------------
    //
    // Updated on the pulse and not on the clock. Between pulses the controller genuinely has no new
    // information, and pretending otherwise by differencing the held value would hand it a stream of
    // zeroes that look like a wheel holding steady.
    if (wheel.pulses != state.lastPulses)
    {
        const auto elapsed = state.sinceUpdate + deltaTime;

        if (state.lastPulses > 0 && elapsed > 0.0)
        {
            state.acceleration = (wheelRoadSpeed - state.lastSpeed) / elapsed;
        }

        state.lastPulses = wheel.pulses;
        state.lastSpeed = wheelRoadSpeed;
        state.sinceUpdate = 0.0;
    }
    else
    {
        state.sinceUpdate += deltaTime;
    }

    const auto request = std::max(requestedPressure, 0.0);

    // Nothing to control: no system, no measurement, or no pedal. The channel falls open and the
    // wheel gets exactly what was asked for — the same expression an unassisted car evaluates, so
    // the two agree to the bit rather than to a tolerance.
    if (!setup.enabled || !wheel.valid || request <= 0.0)
    {
        state.phase = ModulatorPhase::Passive;
        state.pressure = request;
        state.surged = false;

        return request;
    }

    // **Every phase exit is stated in terms the car's own deceleration cannot fake**, and getting
    // that wrong twice is what most of this file's shape is. Both faults were the same mistake: a
    // condition written as though the wheel were the only thing slowing down. During a 1 g stop a
    // *perfectly rolling* wheel reads −9.8 m/s^2 at the rim, so "wait until the wheel accelerates"
    // is a condition that can never fire, and a channel waiting on one sits there with the pressure
    // dumped for the rest of the stop. Measured: a dry 100-0 that should have been untouched came
    // out 22% long with the rear brakes off from a fifth of a second in.
    const auto slip = referenceValid ? estimatedSlip(referenceSpeed, wheelRoadSpeed) : 0.0;

    // Measured against the car's own deceleration where that helps and absolutely where it does not.
    //
    // **The lock trigger is absolute and the recovery triggers are relative**, and that split was
    // measured rather than chosen. A wheel departing is detected by slip; making the *trigger*
    // relative as well puts the whole of the estimator's lag into the one decision that must not
    // fire spuriously, and it fired 119 times on a dry stop where the front wheels physically cannot
    // lock. Recovery is the opposite case: the wheel is coming back towards a car that is still
    // slowing down, so an absolute threshold there is the fault this file already made twice.
    const auto excess = state.acceleration - referenceAcceleration;
    const auto losing = state.acceleration < setup.lockDeceleration;
    const auto surging = excess > setup.recoverySurge;

    // The one judgement the slip-aware recovery law adds, and it is a statement about the ECU's own
    // belief rather than about the tyre: the estimated slip is past the calibrated band. Everything
    // it gates degrades to the previous law when the reference is invalid — and, just as
    // deliberately, when the estimator under-reads, which is what it does in exactly the case the
    // acceleration-only loop exists to survive.
    const auto pastBand = setup.slipAwareRecovery && referenceValid && slip > setup.slipEnter;

    switch (state.phase)
    {
    case ModulatorPhase::Passive:
        state.pressure = request;

        if (losing)
        {
            state.phase = ModulatorPhase::Hold;
        }

        break;

    case ModulatorPhase::Hold:
        // Pressure held while the controller decides whether this is a wheel about to lock or a
        // wheel that has hit something. Slip is what says which, and it is asked every cycle rather
        // than only the first: a dump that fires on deceleration alone fires on excursions that the
        // wheel then recovers from without ever surging, and there is nothing to end it with.
        if (slip > setup.slipEnter)
        {
            state.phase = ModulatorPhase::Dump;
            state.surged = false;
            state.cycles++;
            state.departurePressure = state.pressure;
        }
        else if (!losing)
        {
            state.phase = ModulatorPhase::Reapply;
        }

        break;

    case ModulatorPhase::Dump:
        state.pressure = std::max(0.0, state.pressure - setup.modulator.dumpGradient * deltaTime);

        // **The dump exists to stop the wheel departing, so it ends when the wheel is no longer
        // departing** — not when it has finished coming back. Waiting for a positive surge is waiting
        // for the road to spin the wheel up, and on a slippery surface the road takes tens of
        // milliseconds to do that: at 13/s the caliper is empty by then. Measured that way the
        // pressure went to zero every cycle, which is a third of a second with no brake at all on
        // that corner while the re-apply crawls back, and the pressure cycle came out at 3.6 to
        // 4.8 Hz against a published 4 to 20.
        //
        // **"No longer departing" is not enough on its own while the slip is still past the band**
        // (the slip-aware recovery law, `AntilockSetup::slipAwareRecovery`). A lightly loaded wheel
        // past the tyre's peak reaches an *equilibrium*: road torque nearly balances brake torque on
        // the shallow far side of the curve, so the wheel tracks the decelerating car at a constant
        // 0.4 slip — not departing, not recovering, just stuck — and a dump that ends there hands the
        // pressure straight back to the equilibrium it was called to break. While the estimate says
        // the wheel is still out there, only genuine re-acceleration ends the dump. A wheel actually
        // coming back fires the threshold within milliseconds of the pressure getting low enough, so
        // this does not empty the caliper the way the surge-waiting fault did.
        //
        // **Deeper dumps for deeper departures were built and rejected here, and the measurement is
        // the reason this comment exists** (2026-08-24, late). Scaling the exit threshold from
        // `recoveryAcceleration` at the band's edge to `recoverySurge` a band-width past it — "far
        // past the peak, aggressive dump" in exit terms, no new constant — collapsed the rear axle
        // to 0.47 of its capacity and took the stop from 41.74 m to 45.56: a lightly loaded wheel
        // cannot surge however little brake it carries, so its dumps ran to empty and gave away the
        // falling side of the curve wholesale. The Magic Formula's shallow far side means a wheel
        // held just past its peak keeps delivering; a dump that insists on a convincing recovery
        // trades that force for slip health at three-to-one against. The barely-creeping exit below
        // is not a timidity to fix — it is where the force is.
        if (excess > setup.recoveryAcceleration || (!losing && !pastBand))
        {
            state.phase = ModulatorPhase::Recover;
            state.surged = false;
        }

        break;

    case ModulatorPhase::Recover:
        // Pressure constant while the road spins the wheel back up. A deeply locked wheel surges —
        // hard, because there is a whole tyre's worth of friction and almost no brake — and pressure
        // goes back on when that surge *subsides*, which is the moment the wheel has caught the road
        // rather than the moment it started trying to. A wheel that never surged has already caught
        // it, and waiting for a surge that is not coming is the second half of the fault above.
        state.surged = state.surged || surging;

        // **Back to the dump only if the wheel is departing *again*, not merely still slipping.**
        //
        // Slip alone was the condition until 2026-08-23 and it is the same mistake this file has now
        // made three times: a statement about the wheel written as though the car were not also
        // slowing down. A deeply locked wheel is at 0.65 slip for a tenth of a second *while it comes
        // back*, and on that condition alone the channel ping-pongs Recover -> Dump -> Recover for
        // the whole recovery, taking pressure off on every visit. Traced on a dry full-pedal stop:
        // the front channel reached **zero pressure in 119 ms** and did not get back above a tenth of
        // the request for a third of a second — an anti-lock stop 11% *longer* than locked wheels.
        //
        // It was unreachable before this car had brakes that could lock its front wheels on dry
        // tarmac (`docs/brake-model-brief.md`); every dry measurement of this system was taken on a
        // car whose fronts never left about 0.2 slip.
        //
        // `losing` is what says the wheel is going away again, and it is the same term the entry to
        // `Hold` uses — so a wheel that drags down a second time is dumped exactly as it was the first
        // time, and a wheel that is climbing back at 150 m/s^2 is left alone to climb.
        if (slip > setup.slipEnter && losing)
        {
            state.phase = ModulatorPhase::Dump;
            state.cycles++;
            state.departurePressure = state.pressure;
        }
        else if (pastBand && excess < setup.recoveryAcceleration)
        {
            // **Stuck: neither departing nor coming back, with the slip still past the band.** This
            // is the equilibrium the utilisation instrument found the rear axle living in — the held
            // pressure is the equilibrium pressure, so holding it holds the wheel at three times its
            // peak slip indefinitely, and re-applying is worse. Only a further dump can break it.
            //
            // Not the ping-pong fault this file already made once: that one re-entered the dump on
            // slip alone while the wheel was climbing back at 150 m/s^2, and the `excess` guard here
            // is precisely what it lacked — a wheel genuinely re-accelerating is left alone whatever
            // its slip says.
            //
            // The memory is overwritten with the *stuck* pressure, which is lower than the pressure
            // the wheel originally departed at — conservative on purpose, because the stuck level is
            // the freshest pressure known to be too much for this wheel at this load.
            state.phase = ModulatorPhase::Dump;
            state.cycles++;
            state.departurePressure = state.pressure;
        }
        else if (!(state.surged && surging))
        {
            // Anything below the entry threshold is a wheel worth putting pressure back on, not just
            // anything below the exit threshold — otherwise a channel that settles in the band
            // between the two sits here holding a pressure nobody asked it to hold for the rest of
            // the stop, which is the same stagnation as the fault above wearing different clothes.
            //
            // **Past the band, hold instead** (slip-aware recovery): the wheel is on its way back —
            // the stuck branch above did not fire, so it is re-accelerating — and pressure re-applied
            // now meets it while it is still on the falling side of the curve and puts it straight
            // back down. The road is winning at this pressure or the wheel would not be climbing, so
            // holding cannot stagnate: the slip falls, crosses the band, and the branch below takes
            // over.
            if (!pastBand)
            {
                state.phase = ModulatorPhase::Reapply;
            }
        }

        break;

    case ModulatorPhase::Reapply:
    {
        // **The re-apply is two stages** (slip-aware recovery), which is what a production unit does:
        // the full gradient back up to the pressure the wheel last departed at, then a taper past it
        // that eases to nothing as the estimated slip approaches the band. The first stage is the
        // memory — below a level that provably held this wheel moments ago there is nothing to probe
        // for, and the first cut of this law, which tapered the whole stage, was measured crawling
        // back at 12 bar/s from the bottom of a dump and leaving the rear axle under-braked for
        // tenths of a second per cycle. The second stage is the probe: the closer the estimate says
        // the wheel is to the band, the more gently the pressure goes looking for the peak. Both
        // stages are drawn against the controller's own calibrated band and its own remembered
        // pressure rather than against any new number, and with the law off or the reference invalid
        // the gradient is the previous law's to the bit.
        //
        // A wheel already past the band is never in the fast stage whatever the memory says — that
        // is re-pinning, and it is the one thing the whole law exists to stop.
        const auto proximity =
            setup.slipAwareRecovery && referenceValid
                ? std::clamp((setup.slipEnter - slip) / std::max(setup.slipEnter - setup.slipExit, 1e-9), 0.0, 1.0)
                : 1.0;
        const auto fast = state.pressure < state.departurePressure && !pastBand;

        state.pressure =
            std::min(request, state.pressure + setup.modulator.reapplyGradient * (fast ? 1.0 : proximity) * deltaTime);

        if (losing)
        {
            state.phase = ModulatorPhase::Hold;
        }
        else if (state.pressure >= request)
        {
            // Back to what the driver is asking for with the wheel still turning: the system has
            // nothing left to do and hands the channel back rather than sitting in a control loop
            // that is no longer controlling anything.
            state.phase = ModulatorPhase::Passive;
        }

        break;
    }
    }

    // The unit sits between the master cylinder and the caliper and cannot make pressure the driver
    // is not asking for. Lifting off mid-cycle takes the wheel pressure down with the pedal.
    state.pressure = std::min(state.pressure, request);

    return state.pressure;
}

} // namespace raceengine
