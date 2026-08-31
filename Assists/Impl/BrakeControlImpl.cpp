module;

#include <algorithm>
#include <cmath>
#include <cstddef>

module raceengine.assists;

namespace raceengine
{

double advanceYawMomentDelay(const AntilockSetup& setup, YawMomentDelayState& state,
                             const AntilockChannelState& left, const AntilockChannelState& right,
                             const std::array<double, 2>& frontRequests, const double lateralAcceleration,
                             const bool braking, const double deltaTime)
{
    // Off, not braking, or in a corner. **The corner case is Limpert's own and is not a guard
    // somebody added**: above 0.4 g the delay is switched off deliberately, because there the big
    // braking force at the outer wheel makes a moment that opposes the lateral force's rather than
    // adding to it.
    if (!setup.yawMomentDelay || !braking)
    {
        state = YawMomentDelayState{};
        return 0.0;
    }

    // The start of this brake application, and the cycle counts it began with. **The baseline is the
    // whole reason this works over a session** — see `YawMomentDelayState::baseCycles`.
    if (!state.applied)
    {
        state.applied = true;
        state.baseCycles = {left.cycles, right.cycles};
    }

    // In a corner. Disengages without forgetting the application, because the book switches the
    // feature off here rather than ending it: a car that comes back under 0.4 g in the same stop
    // should still be able to use what is left of its staircase.
    if (std::abs(lateralAcceleration) > setup.yawDelayLateralLimit)
    {
        state.engaged = false;
        state.ceiling = 0.0;
        state.sinceStage = 0.0;
        return 0.0;
    }

    const auto channels = std::array<const AntilockChannelState*, 2>{&left, &right};

    // **"Pressure reduction", not "intervention".** `cycles` counts entries to the dump and nothing
    // else, so it is exactly the book's trigger; `Hold` is a threshold being watched and is not a
    // reduction. Using intervention instead would arm this on a wheel that never let any pressure
    // go.
    const auto reduced =
        std::array<bool, 2>{left.cycles > state.baseCycles[0], right.cycles > state.baseCycles[1]};
    const auto intervening =
        std::array<bool, 2>{left.phase != ModulatorPhase::Passive, right.phase != ModulatorPhase::Passive};

    // Reducing *now*, which is what "as soon as the first pressure reduction ... takes place" says.
    // `cycles` never resets, so a channel that dumped once during an earlier brake application
    // still reports `reduced` on the next one — and arming off that would cap the high wheel from
    // the first millisecond of a stop, before any wheel had gone anywhere, at a ceiling read off a
    // channel whose `pressure` is meaningless while it is passive.
    const auto controlling = std::array<bool, 2>{reduced[0] && intervening[0], reduced[1] && intervening[1]};

    if (!state.engaged)
    {
        // "As soon as the first pressure reduction caused by wheel lockup tendencies takes place at
        // the 'low' wheel." One front having let pressure go while the other has not **is** a split
        // surface as far as this unit can tell, and identifying the high wheel that way needs no
        // surface estimate of any kind.
        if (state.completed || controlling[0] == controlling[1] ||
            (controlling[0] ? reduced[1] : reduced[0]))
        {
            return 0.0;
        }

        const auto high = controlling[0] ? std::size_t{1} : std::size_t{0};
        const auto other = high == 0 ? std::size_t{1} : std::size_t{0};

        // **Where the staircase starts, and this is the one modelling choice in the mechanism.**
        // The book describes two variants: on cars with ordinary handling the high wheel is simply
        // built "in stages or steps", and on cars "with particularly critical handling
        // characteristics" its solenoid is given "a specific pressure-holding and reduction time",
        // with "the pressure modulation of the 'low' and 'high' wheels ... interrelated". This
        // implements the second, because the first is inert here — a driver's pedal is already at
        // everything they are asking for by the time a wheel locks, so a ceiling latched at the
        // request has nothing to stage. **Measured: latched at the request, the delay lasted 17 ms
        // and every apply share gave the same answer.**
        //
        // "Interrelated" read literally: the ceiling is the **low channel's own pressure**, which is
        // the level the road on that side has just proved it will take. It follows that channel down
        // for as long as it is still reducing, and is staged up from wherever it got to once that
        // wheel starts coming back. No new calibrated quantity is introduced anywhere.
        state.engaged = true;
        state.highChannel = high;
        state.ceiling = std::min(frontRequests[high], channels[other]->pressure);
        state.sinceStage = 0.0;

        return state.ceiling;
    }

    // **Handover, and it is the half of the mechanism that keeps the stop short.** "When the
    // pressure at the 'high' wheel has reached its locking level, it is no longer influenced by the
    // signals of the 'low' wheel; that is, it is individually controlled." **Its own first pressure
    // reduction is that**, and it is the same test the trigger uses on the low wheel — deliberately,
    // because "reached its locking level" and "let pressure go" are one event and reading the
    // handover off `Hold` instead ends the delay on a threshold the unit is merely watching.
    // Measured: on `Hold` the staircase never ran and every apply share gave the same answer. The
    // other way out is the staircase arriving: there is then nothing left to hold the wheel back
    // from.
    //
    // **The low wheel going quiet is deliberately NOT a release**, and that was measured: this
    // modulator returns a channel to `Passive` between cycles, so releasing on it ended the delay
    // after 17 ms — one dump-and-recover — and the mechanism did about a third of what it does when
    // it is allowed to run its staircase out.
    const auto low = state.highChannel == 0 ? std::size_t{1} : std::size_t{0};

    if (reduced[state.highChannel] || state.ceiling >= frontRequests[state.highChannel])
    {
        state = YawMomentDelayState{};
        state.completed = true;
        return 0.0;
    }

    // Still reducing at the low wheel: the ceiling follows it down. This is the "pressure-holding
    // and reduction time" half, and it is what stops the high wheel building through the worst of
    // the asymmetry.
    if (channels[low]->phase == ModulatorPhase::Dump)
    {
        state.ceiling = std::min(state.ceiling, channels[low]->pressure);
        state.sinceStage = 0.0;

        return state.ceiling;
    }

    // "Increased in stages or steps": the ceiling holds and then jumps, rather than sliding. The
    // mean rate is a share of the modulator's own re-apply gradient, so the staircase's tread width
    // changes how granular it is and not how fast it gets there.
    state.sinceStage += deltaTime;
    if (state.sinceStage >= setup.yawDelayStagePeriod)
    {
        state.ceiling += setup.yawDelayStagePeriod * setup.modulator.reapplyGradient * setup.yawDelayApplyShare;
        state.sinceStage = 0.0;
    }

    return state.ceiling;
}

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

[[nodiscard]] double advanceAntilockChannel(const AntilockSetup& setup, const BrakeChannel channel,
                                            AntilockChannelState& state, const WheelSpeedReading& wheel,
                                            const double wheelRoadSpeed, const double referenceSpeed,
                                            const double referenceAcceleration, const bool referenceValid,
                                            const double requestedPressure, const double deltaTime)
{
    // The rear outlet is the same valve; the rear INLET is pulsed on its own duty factor in
    // production (the Bosch source at `BrakeModulator::rearReapplyGradient`), and the average of
    // that pulsing is this selection.
    const auto reapplyGradient =
        channel == BrakeChannel::Rear ? setup.modulator.rearReapplyGradient : setup.modulator.reapplyGradient;

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

    // **The slip the slip-aware law reads is computed against a staleness-projected wheel, and
    // only the law's** (2026-08-29, the co-design's first estimator-side step — the "honest fix"
    // the 2026-08-24 entry named, applied to the guards alone). The sensed wheel speed is bounded
    // above by the tooth pitch over the reading's age, which at walking pace understates a rolling
    // wheel by whole tenths of slip — the recorded phantom-dump mechanism. The projection is the
    // ECU's own dead reckoning: the wheel, had it merely followed the car since its last tooth,
    // would be at `lastSpeed + referenceAcceleration · sinceUpdate`; the guard reads whichever of
    // the two stories says the wheel is FASTER, so a genuinely departing or locked wheel (whose
    // projection falls with the car while its pulses stop) keeps its high slip, and a rolling
    // wheel between sparse teeth stops being read as half locked. With fresh pulses the projection
    // equals the held reading and the guard slip IS the sensed slip, so nothing changes at speed.
    // The pre-law transitions — Hold's dump confirm, Recover's re-departure — keep the raw sensed
    // slip: they are the blessed pre-law loop and stay to the bit.
    const auto projectedWheel = std::max(wheelRoadSpeed, state.lastSpeed + referenceAcceleration * state.sinceUpdate);
    const auto guardSlip = referenceValid ? estimatedSlip(referenceSpeed, projectedWheel) : 0.0;

    // The one judgement the slip-aware recovery law adds, and it is a statement about the ECU's own
    // belief rather than about the tyre: the estimated slip is past the calibrated band. Everything
    // it gates degrades to the previous law when the reference is invalid — and, just as
    // deliberately, when the estimator under-reads, which is what it does in exactly the case the
    // acceleration-only loop exists to survive.
    const auto pastBand = setup.slipAwareRecovery && referenceValid && guardSlip > setup.slipEnter;

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
                ? std::clamp((setup.slipEnter - guardSlip) / std::max(setup.slipEnter - setup.slipExit, 1e-9), 0.0, 1.0)
                : 1.0;
        const auto fast = state.pressure < state.departurePressure && !pastBand;

        state.pressure = std::min(request, state.pressure + reapplyGradient * (fast ? 1.0 : proximity) * deltaTime);

        if (losing)
        {
            state.phase = ModulatorPhase::Hold;
        }
        else if (pastBand && excess < -setup.recoveryAcceleration && guardSlip > 2.0 * setup.slipEnter - setup.slipExit)
        {
            // **Stuck in Reapply is a trap without this branch, and it was found locking a front
            // wheel for five seconds** (2026-08-29, the `[.washout-trace]` probe, on the steering
            // pair's own mu 0.35 fixture). Past the band the proximity taper is exactly zero, so
            // this phase holds a constant pressure — a de facto Hold with no exit watch. A wheel
            // that departs SLOWLY under that held pressure never fires `losing` (the excess
            // deceleration stays under the lock threshold on low grip), walks from the band's edge
            // to a full lock over two seconds, and the lock then silences its tone ring — no
            // pulses, so the channel's own acceleration reading freezes and `losing` can never
            // fire again. The ECU watched an estimated slip of 0.998 with a valid reference for
            // 5.5 s and had no transition to take. This is the Recover phase's stuck branch with
            // two more gates, both drawn from what the build measured and neither a new constant.
            // **A full band-width past the band** (2·slipEnter − slipExit): at the band itself —
            // where the Recover branch fires — a re-applying channel is often riding the
            // productive band-edge excursions of a lightly loaded wheel on the shallow far side of
            // its curve, and the first cut of this branch, without the gate, was measured dumping
            // those on the dry rear: law-on utilisation 0.767 → 0.738 and the stop +1.15 m against
            // the law-off arm. A wheel a band-width past the band is not excursioning, it is
            // leaving. **And the wheel must be falling away from the car by more than the
            // calibrated recovery margin** (excess < −recoveryAcceleration — the same half-g every
            // relative trigger in this file keeps above the tone ring's noise floor, used
            // symmetrically): a bare sign test (excess < 0) sat AT that noise floor and collected
            // 6.5 mm of jitter-fired dumps in the dry tail, which was enough to fail the law's own
            // distance gate. The margin leaves a residual mode — a member of the steering
            // ensemble can still ride ~0.4 slip with late rescues — but a LOCK cannot persist:
            // a locked wheel's frozen pulse-derived acceleration reads deeply negative and keeps
            // this branch firing. With the law off or the reference invalid `pastBand` is false
            // and this branch does not exist, which preserves the estimator-collapse degradation
            // to the bit.
            state.phase = ModulatorPhase::Dump;
            state.cycles++;
            state.departurePressure = state.pressure;
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
