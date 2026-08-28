module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
#include <string>
#include <type_traits>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module raceengine.physics:Vehicle;

import raceengine.assists;

import :Ambient;
import :Brakes;
import :Contact;
import :ContactPatch;
import :PhysicsWorld;
import :ProvingGround;
import :RigidBody;
import :Suspension;
import :Telemetry;
import :Tyre;

namespace raceengine
{

// The car, assembled: a chassis that is one rigid body, four corners that each carry one degree of
// freedom, and the forces between them. The tick is a pure function of (setup, state, input, dt) —
// no clock, no globals, no unseeded randomness — which costs nothing now and is the whole of what
// makes a later netcode retrofit possible instead of a rewrite.
//
// **Every number below is a placeholder** for a mid-size car unless a real one has been supplied,
// and is marked where it is stated. They are documented real-world values rather than invented
// ones, so the car behaves plausibly while the model is being validated; replacing them is a data
// change and not a code change.

// A piecewise-linear curve, which is the damper interface the brief asks for rather than a scalar.
// Low and high speed knees are simply more points; a linear damper is two. Nothing here needs to
// change to go from one to the other, which is the point of taking a curve now.
export struct Curve
{
    // Ascending in x. Outside the range the end slope is held rather than extrapolated, because a
    // damper asked about a velocity past its last measured point should be stiff, not wild.
    std::vector<glm::dvec2> points;

    [[nodiscard]] double at(const double x) const
    {
        if (points.empty())
        {
            return 0.0;
        }

        if (points.size() == 1 || x <= points.front().x)
        {
            return points.front().y;
        }

        if (x >= points.back().x)
        {
            return points.back().y;
        }

        for (auto index = std::size_t{1}; index < points.size(); index++)
        {
            if (x <= points[index].x)
            {
                const auto& low = points[index - 1];
                const auto& high = points[index];
                const auto span = high.x - low.x;

                return span > 0.0 ? low.y + (high.y - low.y) * (x - low.x) / span : low.y;
            }
        }

        return points.back().y;
    }
};

// A linear damper stated as a curve, so the common case is not a special case. Bump and rebound
// rates differ on every real damper and are separate arguments here for that reason.
export [[nodiscard]] Curve linearDamper(const double bumpRate, const double reboundRate);

// A wheel-referred kneed damper carried onto the shaft, which is AC's statement turned into this
// model's: `rate` up to `knee` metres per second of *wheel* velocity and `fastRate` above it. The
// shaft moves `ratio` times as fast as the wheel while carrying `1/ratio` of its force, so a
// wheel-referred coefficient divides by the ratio squared and the knee's speed multiplies by it —
// both halves, or the knee lands at the wrong speed. The ratio is the **damper's** and the
// signature says so: a spring's ratio converts a spring rate and nothing else, which is why the
// cross-overload does not exist to call.
export [[nodiscard]] Curve kneedDamper(const double bumpRate, const double fastBumpRate, const double bumpKnee,
                                       const double reboundRate, const double fastReboundRate, const double reboundKnee,
                                       const DamperKinematics& kinematics);
Curve kneedDamper(double, double, double, double, double, double, const SpringKinematics&) = delete;

// A stop that comes in gradually and then very hard. `gap` is how much travel there is before it
// touches; past that the force rises as the deflection to `progression`, so it is soft where it
// first bites and immovable at the end. A linear stop either lets the suspension through it or
// hammers the chassis the instant it touches.
export struct TravelStop
{
    double gap = 0.05;
    double rate = 250000.0;
    double progression = 2.0;

    // Hysteresis, and it is not decoration. A stop modelled as a pure spring returns almost all of
    // what an impact puts into it, so a car dropped onto its stops pogos instead of settling — which
    // is exactly what this model did before the term existed, bouncing higher than it was dropped
    // from. Real bump stops are microcellular elastomer and lose a great deal of it as heat.
    double damping = 30000.0;

    // `past` is how far into the stop the travel has gone, `rate of change` how fast it is going
    // further in. The stop can only push: a damping term large enough to go negative would have it
    // pulling the suspension back into itself as it releases.
    [[nodiscard]] double force(const double past, const double rateOfChange = 0.0) const
    {
        if (past <= 0.0)
        {
            return 0.0;
        }

        const auto elastic = rate * std::pow(past, progression) / std::pow(std::max(gap, 1e-6), progression - 1.0);

        return std::max(0.0, elastic + damping * rateOfChange);
    }
};

export struct CornerSetup
{
    CornerHardpoints hardpoints;

    // N/m along the damper axis, not at the wheel — the motion ratio is what turns one into the
    // other, and it varies with travel, so a wheel rate stated as a constant would be a lie the
    // geometry immediately contradicts.
    double springRate = 55000.0;
    double springFreeLength = 0.0;

    Curve damper = linearDamper(4200.0, 7600.0);

    // Seal and rod friction on the damper shaft, newtons, opposing the shaft's motion at any
    // non-zero velocity and not scaling with it. A `Curve` through the origin is a pure viscous
    // damper and has none; a real one carries a Coulomb term that dominates exactly where the
    // viscous term is smallest — small amplitudes and low velocities, which is straight-line running
    // on coarse tarmac, the first millimetre of a steering input, and the settling after a kerb.
    //
    // **Zero here, and zero on every car in this project, because nobody has sourced one.** Published
    // figures for a passenger car sit around 50 to 200 N at the shaft and a damper dynamometer plot
    // shows it directly as the width of the hysteresis loop at zero velocity — but AC's data has no
    // such number and none was found for this car (docs/suspension-fidelity-brief.md, item 5). The
    // mechanism is here so that the day a plot turns up it is a data change; putting an invented
    // number in would make it an architecture that asserts something nobody measured.
    double damperFriction = 0.0;

    // How fast the shaft has to be moving for the friction term to be fully developed, metres per
    // second. **A numerical choice and not a physical one**: the term is regularised as
    // `friction · tanh(velocity / this)` rather than `friction · sign(velocity)`, because a hard sign
    // term at 360 Hz makes a limit cycle rather than a dead band. Small enough to look like friction,
    // large enough that one tick cannot step across it.
    double damperFrictionSpeed = 0.01;

    TravelStop bumpStop;
    TravelStop droopStop;

    // The tire as a spring in series with the suspension, with the unsprung mass between them —
    // which is what makes wheel hop a mode the model has rather than one it cannot express.
    // 250 kN/m is a documented figure for a passenger radial at road pressure. Placeholder.
    double tireVerticalRate = 250000.0;
    double tireVerticalDamping = 1500.0;

    // **Lateral-force compliance steer**: how far this corner's wheel is twisted about the chassis's
    // up axis per newton of lateral force at its contact patch, radians per newton, **signed**.
    //
    // Every joint in `solveCorner` is an ideal pin or ball, so a rigid linkage takes no toe under
    // load at all. A real one carries rubber, and the toe it takes is not an imperfection — it is
    // *designed in*, because it puts the phase of the steering reaction force ahead of the steering
    // angle and that is most of what the rack feels like. Production cars are set up for a slight
    // **toe-out** under lateral force, which is a negative value here.
    //
    // Only the toe term is carried. A real bush also moves the upright sideways and rearward and
    // takes camber with it, and those have no published figure behind them for this class of car —
    // so the hub stays where the linkage put it. `applyComplianceSteer` is the seam and says why.
    // docs/suspension-fidelity-brief.md, item 1.
    double lateralForceSteer = 0.0;

    // Placeholder: a hub, upright, brake and wheel for a mid-size car.
    double unsprungMass = 38.0;

    // Torsional rate of this corner's half of the anti-roll bar, N·m per radian of *difference*
    // across the axle, expressed at the wheel as N/m of differential travel. Zero disables it.
    //
    // **The number keeps that meaning whether or not the corner states a drop link**, which is the
    // whole of how the geometry was added without moving every car's roll stiffness. A corner with
    // drop-link hardpoints refers this wheel rate onto the link through the link's own motion ratio
    // *at design* — `k_link = k_wheel / ratio²`, the standard referral — so at the design position
    // the bar is worth exactly what it was worth before, and what the geometry buys is that the
    // ratio then varies across the travel and the force rides the link's own Jacobian instead of the
    // wheel's. See `solveAntiRollBar`.
    double antiRollRate = 0.0;

    // Rolling resistance, as a fraction of the vertical load. Applied as a torque on the wheel
    // rather than as a force at the patch, because that is what it physically is: the contact
    // pressure is higher at the leading edge of the patch than the trailing one, and the resultant
    // acts ahead of the wheel centre. Placeholder: 0.012 for a road radial.
    double rollingResistance = 0.012;

    TyreModel tyre;

    // Placeholder: wheel, tire, hub and disc for a mid-size car.
    double wheelInertia = 1.2;

    // What this corner's brake makes at a **fully applied pedal**, N·m — after the hydraulics and
    // after whatever the rear circuit's valve does to them, so it is the number a corner is actually
    // worth rather than one an axle's share has to be applied to.
    //
    // Placeholder here. A car with real hardware derives it (`peakBrakeTorque` in `:Brakes`), which
    // is what took `brakes.ini`'s `MAX_TORQUE` and `FRONT_SHARE` out of the Golf on 2026-08-23; the
    // setup sheet can still override it, because a pad change is a setup change.
    double brakeTorque = 1400.0;

    // The disc as a lump of iron and the pad's fade curve with it — `brakeThermalOf` derives the
    // whole thing from the same `BrakeHardware` that derived the torque above, so nothing here is a
    // second statement of a part. Inert unless `VehicleSetup::brakeThermal` switches it on, and inert
    // whatever that says on a brake whose couple states no fade curve.
    BrakeThermal disc{};

    // The wheel the disc is bolted inside — the third node, and the only path this model has from a
    // brake at 500 °C to a tread at 50. `wheelThermalOf` derives it from a `WheelHardware` and the
    // same `BrakeHardware` the disc came from, so the hat that joins them is stated once.
    //
    // **A default-constructed one has no heat capacity, which means no wheel.** Every step treats
    // that as absent, so a car that does not state a wheel heats exactly as it did before stage 3
    // existed. docs/brake-thermal-brief.md, section 7.
    WheelThermal wheel{};
};

// One aerodynamic surface: a body, a wing, a splitter. Its coefficients are constants this
// milestone — the deferred model replaces each with a lookup against ride height and rake, and
// nothing else moves, because the ride height it would be looked up against is already an input to
// where this is evaluated and the point it acts at is already data.
export struct AeroSurface
{
    // Where it acts, in the chassis frame. Data rather than the centre of gravity, because a
    // splitter and a rear wing pitch the car in opposite directions and that is most of the point.
    glm::dvec3 centre{0.0};

    // Drag area, Cd*A in square metres. Placeholder: 0.30 by 2.2 m2 for a mid-size sedan.
    double dragArea = 0.66;
    // Lift area, Cl*A. Positive lifts, negative presses down. A road sedan makes a little lift.
    double liftArea = 0.10;
};

export struct VehicleSetup
{
    std::array<CornerSetup, cornerCount> corners{};

    // The sprung mass ledger. Chassis, fuel and occupants as separate entries because that is the
    // seam a fuel model needs — mass, centre of gravity and inertia are derived from this every
    // time it changes rather than authored once.
    std::vector<MassComponent> sprung;

    ContactPatchSampling sampling{};

    // The brake system between the pedal and the four corners above.
    //
    // **This is where the other half of the brake bias lives**, and the half a set of calipers cannot
    // express: `CornerSetup::brakeTorque` states what each corner makes at full pedal, and these two
    // state how the pedal gets there — which is not the same shape on both axles once a valve is
    // fitted. Defaults are a car with no servo and no valve, so the pedal maps linearly onto each
    // corner's peak, which is exactly what this model did before either existed.
    BrakeHydraulics brakeHydraulics;
    ProportioningValve rearBrakeValve;

    std::vector<AeroSurface> aero;

    // The body's collision shape and what it does when it hits something. Placeholder: a box a
    // little smaller than a hatchback, so the wheels reach the ground before the bodywork does.
    CollisionBox body;
    ContactMaterial contact;
    // Sea level, 15 degrees. Placeholder, and the one number here a weather model would move.
    double airDensity = 1.225;

    // Steering rack travel per unit of steering input, metres, **signed**: positive demand is a
    // right turn, and whether that needs the rack going positive or negative is a property of the
    // linkage — a steering arm ahead of the kingpin steers the opposite way to one behind it. Every
    // car has to state or derive it; this default is a magnitude with a sign that means nothing, and
    // a setup that leaves it alone has not answered the question.
    double rackTravelPerInput = 0.055;

    // Where this car's ride height is quoted from: the height above the *design contact plane* of
    // the point on the chassis a ride height figure refers to, metres. The body frame puts the
    // design contact plane at y = 0, so this is a ride height at design attitude by construction.
    //
    // **Stated rather than inferred, and the first attempt at inferring it is why.** Taking the
    // underside of the collision box looks principled and is wrong on the one car with real data:
    // AC's `COLLIDER_0` is a coarse body shell 0.32 m up, not a floor, so the Golf reported a 0.30 m
    // ride height — a correct measurement of the wrong surface. A car's ride height is a figure the
    // car states, and a model that derives it from a collision shape derives it from whatever that
    // shape happened to be authored as.
    //
    // Placeholder, like every vehicle figure here, and a candidate for exactly the parameter sweep
    // this file's units are getting.
    double rideHeightReference = 0.12;

    // What the steering *wheel* turns lock to lock, radians. 13.195 rad is 756 degrees, which is
    // 2.1 turns and is this class of car's rack.
    //
    // The vehicle model needs it for exactly one thing and it is not a force: **the telemetry's
    // `Steering Angle` column**, which until 2026-08-21 carried `input.steering` — a dimensionless
    // demand from −1 to 1 — multiplied by 57.2958 and labelled degrees. That is the `Engine RPM`
    // failure exactly: a units conversion applied to a quantity with no units. A demand is not an
    // angle, and turning one into the other needs a number only the steering box has.
    double steeringLockToLock = 13.194689145077131;

    // Whether the tyre's in-plane forces reach the corner's degree of freedom through the linkage's
    // own Jacobian — the geometric load path: roll centre, jacking, anti-dive, anti-squat and
    // anti-lift, which are not four features but four readings of one dot product.
    //
    // Off is the model this project measured everything against: the tyre force reaches the corner
    // as its vertical component times the wheel's vertical Jacobian, so **every newton of load
    // transfer deflects a spring** and none of it travels through the wishbones. That is why the car
    // leans about 12% more than its own springs and bars would make it lean, and why it has exactly
    // zero anti-dive. Total load transfer and longitudinal balance are unaffected either way — those
    // are fixed by the whole-car free body whatever path the force takes inside a corner — so this
    // is an **attitude** switch with a balance side effect, which is the shape the seat reported.
    //
    // A switch rather than a rewrite because the old behaviour is what every measured figure in
    // docs/ was taken under, and because it is the control: `docs/suspension-load-path-brief.md`
    // stage 1 requires both parity gates byte-identical with this off.
    bool geometricLoadPath = false;

    // Whether the wheels' spin is allowed to react on the rest of the car — the driveline's own
    // Newton's third law, in both the places it acts.
    //
    // On the **chassis**: a wheel being spun up or slowed down takes angular momentum from
    // somewhere, and the model gives it none. The chassis receives the tyre force at the contact
    // patch, which is exactly right whenever the wheel's spin is steady, and is short by the wheel's
    // own `I·alpha` whenever it is not — a launch, a shift, a lock-up. Switched on, each corner puts
    // `−I·alpha` back on the body about the wheel's spin axis, which is the term that makes the
    // whole car's angular momentum balance close (docs/suspension-fidelity-brief.md, item 3A).
    //
    // In the **corner**: a chassis-mounted transaxle drives the hub, so the shaft torque does virtual
    // work in the corner's own coordinate as the upright turns about the wheel's spin axis. That term
    // is what converts an outboard brake's contact-patch force line into an inboard drive's
    // wheel-centre line — the textbook distinction, and anti-lift under power on a driven axle. An
    // outboard brake needs no counterpart: its couple is internal to the wheel assembly.
    //
    // Off by default, and off is the model every figure in docs/ was measured under.
    bool drivelineReaction = false;

    // Whether the tyres carry a temperature — the tread's own heat balance, and grip following the
    // tread **core** through the compound's own curve (`TyreThermal`).
    //
    // Off is the tyre this project measured everything against: friction, vertical rate and rolling
    // resistance all constant from the moment the car is built to the end of the session, and a tyre
    // that is therefore always exactly at its best. **Every performance figure in docs/ was taken
    // under that assumption**, and the moment temperature is a state every fixture acquires a
    // precondition it does not state — which is why the default seeds at
    // `tyreDefaultTemperature`, the middle of the curve's flat plateau, so that switching this on
    // changes nothing until something heats or cools a tyre.
    //
    // A switch rather than a rewrite for `geometricLoadPath`'s reason exactly: off is the control,
    // and `OSR_TYRE_THERMAL=off` is the way back from the seat without a rebuild.
    // docs/tyre-state-brief.md.
    bool tyreThermal = false;

    // Whether the brake discs carry a temperature, and the pads' friction follows it — which is fade.
    //
    // Off is the brake every figure in docs/ was measured on: a coefficient that is the same on the
    // first stop of the day and the tenth in a row. **The inertness proof comes from the opposite end
    // of the curve to the tyre's** and that is worth noticing rather than copying blindly — a pad's
    // rated friction is flat *cold*, so seeding a disc at ambient is both the physical seed and the
    // one that changes nothing, where the tyre had to be seeded warm to sit on its plateau.
    //
    // `OSR_BRAKE_THERMAL=off` is the control and the way back. docs/brake-thermal-brief.md.
    bool brakeThermal = false;

    [[nodiscard]] double unsprungMass() const
    {
        auto total = 0.0;
        for (const auto& corner : corners)
        {
            total += corner.unsprungMass;
        }

        return total;
    }
};

// The wheel inertias in corner order, from the one place they are stated. The driveline needs them
// to refer an axle's inertia through the gearing, and a caller assembling that array by hand is a
// second statement of a number that already has an owner — which is what it was, and nothing made
// the two agree.
export [[nodiscard]] std::array<double, cornerCount> wheelInertias(const VehicleSetup& setup);

// What the caliper at each corner is at, pascals, for a given brake pedal. The master cylinder's own
// characteristic for the front pair and the proportioning valve's output for the rear.
//
// It is a function of the *car* rather than of a corner because the two axles share one master
// cylinder, which is the whole reason a fixed bias cannot be right everywhere.
export [[nodiscard]] std::array<double, cornerCount> brakeCircuitPressures(const VehicleSetup& setup,
                                                                           const double pedal);

// The fraction of `CornerSetup::brakeTorque` a corner makes at a given pedal — that corner's pressure
// against its own pressure at a fully applied pedal.
//
// **This is what makes the bias pedal-dependent instead of a constant ratio.** Both axles are linear
// in it below the valve's knee and the servo's runout; above either, they are not, and they are not
// in the same way. With no servo and no valve it is the pedal itself, to the bit.
export [[nodiscard]] double brakePedalResponse(const VehicleSetup& setup, const std::size_t corner, const double pedal);

export struct CornerState
{
    // The corner's whole degree of freedom: where the lower wishbone is, and how fast it is moving.
    double wishboneAngle = 0.0;
    double wishboneRate = 0.0;
    // Wheel spin, integrated in the same loop as everything else. Nothing drives it yet — the road
    // does, through the tire, which is enough to roll and to brake.
    double wheelSpeed = 0.0;

    // The carcass deflections, which are what make the tire transient rather than instantaneous.
    TyreState tyre;

    // How far the bushes have twisted this corner about the chassis's up axis, radians. State rather
    // than a derived quantity, because it is driven by the tyre force and the tyre force is driven by
    // the toe: solving that inside one tick means an iteration, and this model's only iteration is
    // the contact patch's.
    //
    // Carried from the previous tick instead, which is one tick of explicit lag at 360 Hz. **That was
    // refused for the jacking force and is accepted here, and the difference is the sign of the
    // loop**: more toe-out lowers the slip angle, which lowers the force, which lowers the toe-out,
    // so the lag sits inside negative feedback and settles. Jacking is the other way round. A rubber
    // bush also has a real relaxation of about this order, so what the lag models is not nothing —
    // but nobody has sourced its time constant, so it is not claimed as one.
    double complianceSteer = 0.0;

    // The brake disc's temperature, degrees Celsius. Seeded at `brakeDefaultTemperature`, which is
    // cold — see `VehicleSetup::brakeThermal` for why cold is the inert seed here and warm was the
    // inert seed for the tyre.
    double discTemperature = brakeDefaultTemperature;

    // The wheel's, the same way. **A disc temperature without its wheel is not a state**, which is
    // why `seedDiscTemperatures` sets both: a fixture measuring a first stop of the day is asserting
    // that the whole corner is cold, and a wheel left at whatever the last run finished on would
    // carry a stint's worth of heat into it.
    double wheelTemperature = brakeDefaultTemperature;
};

// The vehicle's own state and nothing else's. Engine speed lived here until the driveline was given
// a state of its own: it was written by `stepDriveline`, read by nobody in this file but the
// telemetry fill, and its being here made `:Vehicle` the keeper of a number belonging to a partition
// it does not import. `DrivelineState` in `:Driveline` owns it now, and the caller that steps both
// is what joins them.
export struct VehicleState
{
    RigidBodyState chassis;
    std::array<CornerState, cornerCount> corners{};
};

// Put every one of a car's brake discs at a stated temperature. A fixture measuring a tenth stop in a
// row starts where the ninth left off; one measuring a first stop of the day says so with this.
export void seedDiscTemperatures(VehicleState& state, const double celsius);

// Put every one of a car's tyres at a stated temperature. What a fixture calls when it wants a cold
// car — and the reason it is spelled at the call rather than defaulted is that a starting temperature
// is part of a measurement: a stop from 20 °C and a stop from 85 °C are different experiments, and a
// fixture that does not say which one it is running has not stated its own preconditions.
export void seedTyreTemperatures(VehicleState& state, const double celsius);

// `raceengine.assists` restates its own wheel count rather than importing this one, because it must
// not be able to import anything from this module — see the head of `Assists/Api/WheelSensors.cppm`.
// This is the join: two statements of one number that the compiler will not let disagree.
static_assert(wheelCount == cornerCount, "the assist layer and the vehicle must agree on how many wheels a car has");

static_assert(std::is_trivially_copyable_v<VehicleState>, "the harness saves and restores this by copying its bytes");
static_assert(std::is_standard_layout_v<VehicleState>, "and rollback will later");

export struct VehicleInput
{
    // -1 to 1, and 0 to 1. Sampled once per physics tick and never polled from inside the model.
    double steering = 0.0;
    double throttle = 0.0;
    double brake = 0.0;
    // 0 with a foot off it and 1 fully depressed, which is the way round the pedal works: pressing
    // it *releases* the clutch. The auto-clutch layer fills it in when nobody is on it, and a driver
    // past the pedal's free play takes it back outright.
    double clutch = 0.0;
    std::int32_t gear = 0;
};

// Nothing driving the wheels: a car being pushed, coasted or dropped, and every tick taken before a
// driveline exists. Named rather than written `{}` at the call, because a bare brace there reads as
// an omission and this is a statement.
export inline constexpr std::array<double, cornerCount> noDriveTorque{};

export struct CornerForces
{
    double spring = 0.0;
    double damper = 0.0;
    double bumpStop = 0.0;
    double droopStop = 0.0;

    // **Always the equivalent force at the wheel**, newtons, whichever bar model this corner is on —
    // negative on the wheel that is further into bump, because the bar pushes it back out. A corner
    // that states a drop link computes its force along the link and reports the wheel force that
    // would do the same virtual work, so this field means one thing and telemetry reading it does
    // not have to know which model a car is on.
    double antiRoll = 0.0;

    // And the force along the drop link itself, newtons, zero on a corner that states none. The two
    // corners of an axle carry this exactly equal and opposite — a torsion bar is one internal force
    // — which is **not** true of `antiRoll` above once the link has a motion ratio, because the two
    // corners sit at different points in their travel and so convert it differently.
    double antiRollLink = 0.0;

    double tireVertical = 0.0;
};

// The tire's frame at the contact patch, and what it is doing in it. Kept so a test can assert on
// the slip a wheel is working at rather than only on the force that came out of it.
export struct WheelContact
{
    glm::dvec3 forward{0.0, 0.0, 1.0};
    glm::dvec3 lateral{1.0, 0.0, 0.0};
    double longitudinalVelocity = 0.0;
    double lateralVelocity = 0.0;
    double effectiveRadius = 0.0;
    TyreSlip slip;
    TyreForces tyre;
};

// Everything the tick worked out about one corner, kept so the caller can fill telemetry and the
// tests can assert on the parts rather than only on the whole.
export struct CornerSolution
{
    SuspensionState suspension;
    ContactPatch patch;
    CornerForces forces;
    WheelContact contact;
    double generalisedForce = 0.0;
    double generalisedInertia = 0.0;
    double damperVelocity = 0.0;
};

// How high the car's floor is above the road, per corner and as the two figures a rake-sensitive
// map is looked up against.
//
// **The datum is `VehicleSetup::rideHeightReference`**, which the car states — see there for why it
// is stated and not read off the collision box. Each corner's is measured perpendicular to that
// corner's *own* contact patch, so a car on a banked road reports the clearance it actually has
// rather than a world-vertical drop.
//
// The version this replaces was `state.chassis.position.y - patch.centre.y` averaged over the four
// corners and then discarded — three separate faults in one expression. The centre of mass is not
// the floor, so it measured about 0.5 m where a ride height is about 0.12; a world-vertical
// difference is not a clearance on any road that is not level; and averaging four corners into one
// scalar destroys rake, which is the entire difference a rake-sensitive map exists to express.
//
// Nothing consumes this yet, and that is deliberate: the aero it is for is out of scope. What was
// not acceptable was leaving a *wrong* seam in place for that work to be built on.
export struct RideHeight
{
    std::array<double, cornerCount> corners{};

    // The axle means, and the difference between them. Positive rake is the rear sitting higher than
    // the front, which is the sense every aero map is drawn in.
    double front = 0.0;
    double rear = 0.0;
    double rake = 0.0;

    // How many corners were touching. A corner in the air is measured against the plane the others
    // define, which is the best that can be said about it; with none touching there is no road to
    // measure from and every figure above is zero.
    std::uint32_t grounded = 0;
};

export struct VehicleStep
{
    std::array<CornerSolution, cornerCount> corners{};
    RideHeight rideHeight;
    // What the bodywork was touching this tick, with the impulses the solver settled on. Kept so a
    // test can assert on the contact rather than only on where the car ended up.
    ContactManifold contacts;
    TelemetryFrame telemetry;
};

// What the road put on each wheel this tick — the tire's own reaction, from the one place it is
// computed. Positive drives the wheel forward, so a rolling wheel's is negative.
//
// The driveline needs it, and passing it rather than letting the driveline guess is the same rule
// `wheelInertias` follows. A coupling's constraint torque is `(Id*Te - Ie*Td)/(Ie+Id)`, and in first
// gear the engine's reflected inertia is twelve times the wheels', so `Td` is most of the answer: the
// term the engine's own torque contributes is 7.6% of it. Told `Td = 0`, the only thing left in the
// constraint's arithmetic that can make up the difference is a speed difference, and a *locked*
// clutch then carries a steady slip — measured at 18 rad/s in first gear, 170 rpm of engine speed
// that should not be there.
export [[nodiscard]] std::array<double, cornerCount> roadTorques(const VehicleStep& step);

// The spring free length that puts a corner in equilibrium at its design position under a given
// sprung load. Not required — a car settles wherever its springs put it — but a setup whose design
// position is also its static position is the one whose camber and roll centre curves mean what
// they say, and computing it beats guessing at it. Solved on the **spring's own element**
// (`springElementOf`) since the spring migration; on a coaxial car that is the damper's element
// and the answer is bit-identical to what it always was.
export [[nodiscard]] std::expected<double, std::string> springFreeLengthForLoad(const CornerSetup& corner,
                                                                                const double sprungLoad);

// The spring's own solution at a solved suspension position: its element's length and Jacobian
// beside the force that length produces. This is what the force pass consumes since the spring
// migrated onto its own element — and the seam a test reaches the production spring arithmetic
// through without standing up a whole vehicle. Closed form, cannot fail.
export struct SpringSolution
{
    double length = 0.0;
    double lengthPerAngle = 0.0;
    double force = 0.0;
};

export [[nodiscard]] SpringSolution solveSpringForce(const CornerSetup& corner, const SuspensionState& suspension);

// The damper's own geometry at a solved position, from the element evaluator — the damper-side
// counterpart of `SpringSolution`, and since step 7 the source of the production damper velocity.
// Deliberately geometry only: the damper force, the stops and the implicit damping still read the
// solver's state fields, and their migration is a later step's. On every current car the element
// is the solver's own damper, so length and Jacobian are the state's values bit for bit.
export struct DamperSolution
{
    double length = 0.0;
    double lengthPerAngle = 0.0;
};

export [[nodiscard]] DamperSolution solveDamperGeometry(const CornerSetup& corner, const SuspensionState& suspension);

// The damper's force at a solved position and wishbone rate — the damper-side counterpart of
// `SpringSolution`'s force, and since step 8 what the force pass consumes. The force law is
// untouched: the same curve, the same compression-positive velocity `−lengthPerAngle · q̇`, the
// same signs; only the geometry source is the damper's own element. The stops and the implicit
// damping deliberately still read the solver's state fields.
export struct DamperForceSolution
{
    double length = 0.0;
    double lengthPerAngle = 0.0;
    double velocity = 0.0;
    double force = 0.0;
};

export [[nodiscard]] DamperForceSolution solveDamperForce(const CornerSetup& corner, const SuspensionState& suspension,
                                                          const double wishboneRate);

// The damper's contribution to the corner's implicit damping, as a coefficient rather than a
// force: the damper's own Jacobian squared times the curve's local slope at the current shaft
// velocity, floored at zero. The law is untouched since before the element migrations — this is
// the same expression the integration always solved against, now stated on `DamperForceSolution`
// so the coefficient reads the damper's element and a test can hold it to the old bits.
export [[nodiscard]] double damperDampingCoefficient(const CornerSetup& corner, const DamperForceSolution& damper);

// The anti-roll bar at one corner: its force, and the Jacobian that force rides.
//
// Two models behind one function, chosen by the data (`dropLinkStated`). A corner with no drop-link
// hardpoints gets the wheel-referred bar this model has always had — a rate times the difference in
// *wheel travel*, on the wheel's own vertical Jacobian, arithmetic for arithmetic. A corner that
// states one gets the bar on its own element: the rate referred to the link through the link's design
// motion ratio, the force from the difference in *link* displacement, and the projection the link's
// own. Both report a wheel-equivalent force so that `CornerForces::antiRoll` means one thing.
//
// The referral uses **both** corners' design ratios, so the pair's link forces are exactly equal and
// opposite even if an axle is not mirrored — which is a stronger statement than the wheel-referred
// model could make.
export struct AntiRollBarSolution
{
    // Along the drop link, newtons, positive pushing its ends apart. Zero without a stated link.
    double linkForce = 0.0;
    // d(link length)/dWishboneAngle, the Jacobian `linkForce` acts through. Zero without one.
    double lengthPerAngle = 0.0;
    // The equivalent force at the wheel, which is what the corner's generalised force uses in the
    // wheel-referred model and what both models report.
    double wheelForce = 0.0;
    // Which of the two models answered.
    bool geometric = false;
};

export [[nodiscard]] AntiRollBarSolution solveAntiRollBar(const CornerSetup& corner, const SuspensionState& suspension,
                                                          const CornerSetup& across,
                                                          const SuspensionState& acrossSuspension);

// The damper shaft's displacement from its design length, positive in bump — the whole of what
// the bump and droop stops measure their gaps against, because a real stop lives on the shaft.
// Since step 12 both lengths are the damper element's: the design length evaluated at q = 0 and
// the current length off the same element, so the stops' kinematics follow the damper wherever
// its geometry goes. The stop force law and the stops' generalised contribution are untouched.
export [[nodiscard]] double damperShaftCompression(const CornerSetup& corner, const DamperForceSolution& damper);

// Standard gravity, and the only constant in this file that is not a placeholder.
inline constexpr auto earthGravity = 9.80665;

// What the outside front tyre carries when this car is cornering at its own limit, newtons.
//
// **A fixed point rather than a chosen manoeuvre**, which is what makes it a property of the car
// instead of a number somebody drove to. Lateral load transfer is `m_axle · a_y · h / t`, and the
// lateral acceleration the car can hold is set by the friction the outside tyre has — which falls
// with the very load the transfer is putting on it. So the two define each other, and the honest
// statement is the load that satisfies both:
//
//     Fz = W_front/2 + mu(Fz) · m_front · g · h / t_front
//
// It converges in three or four passes from the static load because tyre load sensitivity is weak
// (this car's exponent is 0.1926) and the map is a strong contraction.
//
// **Validated against a real lap.** For the Golf this returns 6375 N on the outside front, against a
// median 5851 N measured on Dominic's 201-second Bathurst session through the slip band where the
// aligning moment peaks — **9% high**, and high is the expected direction: the approximation here is
// that the front axle carries its own transfer with the whole car's centre of gravity height, rather
// than a share set by the front/rear roll-stiffness split, and it takes no account of a driver who
// is not at the limit on every corner. What is being placed against it is a knee and a taper, so
// nine percent is comfortably inside what the answer needs to be worth.
//
// The rack force that comes out the far end of it agrees better than the load does: 1743 N derived
// against 1617 N measured, 7.8%, because the geometry between them is exact and only the load is
// modelled.
export struct SteeringLimitLoads
{
    // What one front wheel carries with the car at rest, newtons.
    double staticPerWheel = 0.0;
    // And what the outside and inside front carry at the limit. They sum to twice the static one:
    // lateral transfer moves load across the axle, it does not add any.
    double outside = 0.0;
    double inside = 0.0;
};

export [[nodiscard]] SteeringLimitLoads steeringLimitLoad(const VehicleSetup& setup);

// Placeholder geometry for one corner, mirrored by side and placed at an axle station. Short-long
// arm, so camber gain comes out of the linkage rather than being asked for.
export [[nodiscard]] CornerHardpoints placeholderCorner(const CornerSide side, const double axleZ);

// Refuses a corner whose stops cannot do their job, which is a different question from whether its
// geometry is sound and is not covered by `validateCorner`.
//
// The failure this exists for is worth stating, because it does not look like a suspension problem
// at all. A stop whose gap is larger than the shaft's travel never engages; the corner runs instead
// to the linkage's geometric limit, where it is held by a clamp — and a clamp is a constraint with
// no reaction force. The spring goes on pushing the chassis up with nothing pushing back, so a car
// in the air gains energy every tick, pitches onto its nose and eventually leaves the world. It
// reads as an integrator fault or a contact fault and is neither.
export [[nodiscard]] std::expected<void, std::string> validateCornerSetup(const CornerSetup& corner);

// **A placeholder car**, and every number in it is one: a 1352 kg mid-size sedan, 2.7 m wheelbase,
// 1.44 m track, 60/40 front weight distribution, ride frequencies of about 1.2 Hz front and 1.3 Hz
// rear. They are documented real-world figures for the class rather than invented ones, so the model
// behaves plausibly while it is being validated, and swapping in a measured car is a change to this
// function and to nothing else.
export [[nodiscard]] std::expected<VehicleSetup, std::string> placeholderSedan();

// One tick of one vehicle. Pure in (setup, state, input, driveTorques, dt) and the world it is
// standing on: no clock is read, no global is touched, nothing is random. The structure is the
// architecture the brief asks for and everything later plugs into it — clear the accumulators, let
// every element contribute a force, then integrate once at the bottom. No element writes a velocity
// or a position.
//
// `driveTorques` is what the driveline is putting on each wheel, in newton metres, and it arrives as
// its own argument rather than as a field on `VehicleInput` deliberately. `VehicleInput` is the
// driver's — steering, throttle, brake, gear — and is the packet a rollback netcode would transmit
// and replay. Driveline torque is *derived* from state and must be recomputed on a replay rather
// than trusted from the wire, so it has no business in the struct that gets sent.
//
// `brakes` is the assist layer's output and follows the same rule for the same reason. Defaulted,
// and the default is not "no brakes" — it is **"nobody intervened"**, under which this function
// applies the driver's demand against each corner's own peak exactly as it did before an assist
// layer existed. That is what lets every fixture in the suite go on calling this with six arguments
// and get an answer identical to the bit rather than merely close, which is the only kind of
// regression evidence worth having for a car underneath a controller.
//
// `ambient` is the weather, and it follows the same rule for the same reason: it is a property of
// the *scene* rather than of the car, it is read by exactly one thing — the tread's heat balance —
// and it is defaulted so that every fixture that does not care goes on calling this with six or
// seven arguments. A car with `tyreThermal` off never reads it at all.
export [[nodiscard]] std::expected<VehicleStep, std::string>
stepVehicle(const VehicleSetup& setup, VehicleState& state, const VehicleInput& input,
            const std::array<double, cornerCount>& driveTorques, const PhysicsWorld& world, const double deltaTime,
            const BrakeCommand& brakes = {}, const AmbientConditions& ambient = {});

} // namespace raceengine
