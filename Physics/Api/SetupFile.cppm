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

import raceengine.assists;

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

    // Seal and rod friction on the damper shaft, newtons. A damper setting, so it belongs on the
    // sheet beside the bump and rebound rates it sits on top of.
    std::optional<double> damperFriction;

    // The **bump** stop's viscous damping, N·s/m on the shaft, and its rate-independent hysteresis
    // as a fraction of its own elastic force (`TravelStop::hysteresis` carries the sourcing). On the
    // sheet because the shipped 40000 N·s/m is a placed number — about five times the front
    // corner's own critical damping — and the sourced alternative (`stopdamping 0` with
    // `stophysteresis 0.07`) is a feel change that belongs to whoever is driving. The droop stop is
    // deliberately not reachable from here: on a strut it is the damper topping out, a different
    // mechanism with its own account (docs/known-red.md, the droop-travel entry).
    std::optional<double> stopDamping;
    std::optional<double> stopHysteresis;

    // `stopdynamic 1` installs the sourced jounce-bumper branch on this axle's bump stops —
    // `jounceBumperCandidate`, a modified Dahl friction element plus five Maxwell elements
    // transferred from Pech et al.'s measured specimen onto this stop's own static law. Anything
    // else, including the absent key, leaves the stop exactly as the car states it.
    //
    // It is one key rather than twenty because there is nothing here for a driver to tune: the
    // parameter set is derived from the stop it is installed on, and the two numbers inside it that
    // the source does not publish are placed and flagged at their own fields. The A/B this exists
    // for is `front.stopdamping 0` with `front.stopdynamic 1` — the sourced rate-dependent branch
    // against the placed viscous constant that stands in for it today.
    //
    // **Nothing on any car states it and no seat verdict exists on it.**
    std::optional<double> stopDynamic;

    // Lateral-force compliance steer, **degrees per kilonewton** — the unit a K&C rig reports it in
    // and the unit the published figures behind it are quoted in, converted on the way in rather
    // than making a human write radians per newton. Negative is toe-out, which is what a production
    // car is set up for.
    std::optional<double> complianceSteer;

    // Lateral-force compliance camber, **degrees per kilonewton**, the same rig and the same unit.
    // Positive means the patch complies with the force and the tyre leans over it, which is what
    // every production car measured does; `compliancecamber 0` is the A/B against the car's own
    // stated figure.
    std::optional<double> complianceCamber;

    // Longitudinal recession, **millimetres per kilonewton** of longitudinal force at the patch —
    // the unit the front axle's design band is quoted in (Heissing/Ersoy Table 1-6: front
    // 4–8 mm/kN of braking force). Positive means the wheel complies with the force: rearward
    // under braking. The Golf states 6 front / 10 rear since 2026-08-29 night (a seat-accepted
    // design target — the grade is flagged where it is stated), so **`recession 0` is the A/B and
    // the way back**. The rear's published band is 8–16 mm PER G of deceleration, a different
    // unit; the worked conversion is beside the Golf's compliance figures in
    // `PublishedCarsImpl.cpp`.
    std::optional<double> recession;
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

// The pedal motors' cue thresholds, as multiples of the tyre's own peak slip — the range stage one
// of the pedal feedback renders lockup and wheelspin over. On the sheet for the reason `ffb.gain`
// is: how sensitive a cue should be is a feel number that belongs to whoever is driving, and a feel
// number that needs a rebuild is one that never gets found. The two full points are separate because
// the quantity is not symmetric — slip is bounded at −1 under braking and unbounded under power.
//
// Applied by the game beside the force-feedback tune rather than by an overload here, because the
// consumer (`PedalFeedbackSetup`) lives in the input layer, which this module cannot name.
export struct PedalTune
{
    std::optional<double> onsetPeaks;
    std::optional<double> brakeFullPeaks;
    std::optional<double> throttleFullPeaks;
};

// The car's electronics, as three lines a driver can change between laps.
//
// **On the sheet rather than in the environment, and that is where they belong.** Which assists are
// on is a driver's setting in exactly the way `ffb.gain` and the pedal cue are: it is felt, it is
// argued about, and a setting that needs an environment variable is one nobody changes twice.
//
// Optional like everything else here, and absent means **off**, because that is what
// `golfGtiMk7Assists` builds and a setup file is an overlay on a car rather than a car. A real Mk7
// GTI leaves the factory with all three on; making that the default is a decision with a golden
// re-bless attached, not a line in a parser.
export struct AssistTune
{
    std::optional<bool> antilock;
    std::optional<TractionMode> traction;
    std::optional<bool> cornering;

    // `assist.yawdelay 0|1` — yaw moment build-up delay, which on a split-friction surface builds
    // the high-grip front wheel's pressure in stages from the moment the low-grip one first lets
    // pressure go (`AntilockSetup::yawMomentDelay` carries Limpert's passage). **Off on the car and
    // off here.** It is on the sheet rather than in car data because the book is explicit that the
    // feature is "a compromise between good steering response and minimized stopping distance" and
    // that manufacturers differ — which makes it a driver's setting, like `assist.abs` beside it.
    std::optional<bool> yawDelay;

    // `assist.yawdelayshare 0.5` — how much of the modulator's own re-apply gradient the staged
    // build climbs at. **Placed, not sourced**, and it matters more than any other number in the
    // feature: measured on a split surface at a realistic pedal application, 0.10 costs 76% of the
    // stopping distance and 0.50 costs 0.9%, for much the same steering benefit (`[.yaw-delay]`).
    // On the sheet so that the seat can find its own answer without a rebuild.
    std::optional<double> yawDelayShare;
};

export struct VehicleTune
{
    AxleTune front;
    AxleTune rear;
    DifferentialTune differential;
    FeedbackTune feedback;
    SteeringTune steering;
    PedalTune pedal;
    AssistTune assists;
};

// One tune, from the text of a file. Line oriented, `key value`, `#` to the end of the line is a
// comment, blank lines ignored.
//
// **An unknown key is an error rather than a line to skip**, which is the whole difference between a
// setup file and a wish. A misspelled `front.sping` that is quietly ignored is a change the driver
// made, felt nothing from, and then spent the afternoon compensating for elsewhere — the same shape
// as the inert `peakSlipScale` this codebase already caught once.
export [[nodiscard]] std::expected<VehicleTune, std::string> parseVehicleTune(const std::string_view text);

// Whether this sheet says anything at all. A file of nothing but comments is a valid setup and is
// what ships — so a game can tell "the driver changed something" from "there is a setup file", and
// not announce a reload that changed no number on the car.
export [[nodiscard]] bool statesAnything(const VehicleTune& tune);

// The tune onto a car. Absent fields leave what the car said, which is what makes a setup file that
// states one number a setup file that changes one thing.
//
// **The car must be freshly built, not the one already being driven.** An overlay applied on top of
// itself is one that can only ever add: delete a line, save, and the value it used to state is still
// in the car with nothing left in the file to explain it. That is the same failure the unknown-key
// rule exists to prevent — a change the driver made and felt nothing from — and it is also what makes
// `steering.invert` expressible as the flip it actually is rather than as a claim about a sign.
export void applyVehicleTune(const VehicleTune& tune, VehicleSetup& vehicle);

// And onto the driveline, which owns the one thing on a setup sheet that is not a corner.
export void applyVehicleTune(const VehicleTune& tune, DrivelineSetup& driveline);

// And onto the electronics. An overload here rather than in the game — unlike `PedalTune`, whose
// consumer lives in the input layer this module cannot name — because `raceengine.physics` imports
// `raceengine.assists` and can therefore say what an `AssistSetup` is.
//
// **The car must be freshly built here too**, for the reason stated above: an overlay applied on top
// of itself can only ever add, so deleting `assist.tc` from the sheet would leave traction control
// on with nothing in the file to explain it.
export void applyVehicleTune(const VehicleTune& tune, AssistSetup& assists);

} // namespace raceengine
