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

    TravelStop bumpStop;
    TravelStop droopStop;

    // The tire as a spring in series with the suspension, with the unsprung mass between them —
    // which is what makes wheel hop a mode the model has rather than one it cannot express.
    // 250 kN/m is a documented figure for a passenger radial at road pressure. Placeholder.
    double tireVerticalRate = 250000.0;
    double tireVerticalDamping = 1500.0;

    // Placeholder: a hub, upright, brake and wheel for a mid-size car.
    double unsprungMass = 38.0;

    // Torsional rate of this corner's half of the anti-roll bar, N·m per radian of *difference*
    // across the axle, expressed at the wheel as N/m of differential travel. Zero disables it.
    double antiRollRate = 0.0;

    // Rolling resistance, as a fraction of the vertical load. Applied as a torque on the wheel
    // rather than as a force at the patch, because that is what it physically is: the contact
    // pressure is higher at the leading edge of the patch than the trailing one, and the resultant
    // acts ahead of the wheel centre. Placeholder: 0.012 for a road radial.
    double rollingResistance = 0.012;

    TyreModel tyre;

    // Placeholder: wheel, tire, hub and disc for a mid-size car.
    double wheelInertia = 1.2;
    // Peak brake torque at this corner, N·m. Placeholder, and split front/rear by the setup.
    double brakeTorque = 1400.0;
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
    double antiRoll = 0.0;
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
// they say, and computing it beats guessing at it.
export [[nodiscard]] std::expected<double, std::string> springFreeLengthForLoad(const CornerSetup& corner,
                                                                                const double sprungLoad);

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
export [[nodiscard]] std::expected<VehicleStep, std::string>
stepVehicle(const VehicleSetup& setup, VehicleState& state, const VehicleInput& input,
            const std::array<double, cornerCount>& driveTorques, const PhysicsWorld& world, const double deltaTime);

} // namespace raceengine
