module;

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module raceengine.traffic:Dynamics;

import raceengine.physics;

namespace raceengine
{

// A traffic car that has been hit, as a body rather than as a point on a lane.
//
// **This is deliberately not `stepVehicle`, and the reason is what a disturbed traffic car is for.**
// The full vehicle model is 21 rays a wheel of contact patch, a tread heat balance, a disc heat
// balance, a driveline, brake hydraulics and a linkage solved twice a tick — about 18 microseconds
// of a 2.78 ms budget, per car. All of that exists to answer questions about how *the player's* car
// behaves at the limit, and none of those questions is asked of a hatchback that has just been
// nudged into a bus lane. What is asked of one is: does it stay on its wheels, does it slide the way
// a car slides, and can it steer itself back onto its lane or over to the kerb. A rigid body on four
// spring rays with a friction-circle tyre answers all three at about a tenth of the cost, which is
// the difference between eight disturbed cars and thirty.
//
// It is a *simple* model and not a *wrong* one: the load transfer is real, because the forces are
// applied where they act; the friction circle is real, so a car cannot brake and corner at once past
// its grip; and the tyre saturates rather than growing without bound. What it does not have is any
// of the fidelity the seat is judged on, which is exactly the fidelity nobody is judging here.
//
// Metres, seconds, kilograms, newtons, radians.

export inline constexpr std::size_t simpleWheelCount = 4;

// Front left, front right, rear left, rear right. The front pair steers.
export enum class SimpleWheel : std::uint8_t
{
    FrontLeft,
    FrontRight,
    RearLeft,
    RearRight
};

export struct SimpleVehicleSetup
{
    // **Measured off the fleet rather than chosen.** The five cars exported for Grand City Parkway —
    // a Commodore, a Carnival, a Legacy, a Hilux and a Charger — are 4.71 to 5.19 m long, 1.99 to
    // 2.23 m wide over their mirrors and 1.78 to 2.11 m tall, on wheelbases of 2.91 to 3.05 m. One
    // body serves all of them because they are traffic; what it must not be is the 4.3 m hatchback
    // this defaulted to before there was a fleet, which left every car in the city colliding with a
    // box three quarters of a metre shorter than it looked.
    double massKilograms = 1750.0;

    // The bodywork's collision box, in the body frame, whose origin is on the road between the axles
    // — the same convention the full vehicle model uses, and the one every car in the fleet is
    // authored to. Narrower than the mirrors and shorter than the roof: what collides is the body.
    //
    // **The floor is 0.30 m over the road** (2026-09-11; it was 0.02). Grand City Parkway's kerbs are
    // 0.15 m faces, and a box that reaches the road meets every one of them as a wall: traffic on
    // its lanes never did, and the police leave the lanes on every route that crosses a kerb, and
    // stopped dead on it. The same fault, and the same placement, as the Charger's on 2026-09-09
    // (docs/police-pursuit-brief.md §9, its floor 0.31): a kerb reaches the wheels, whose rays climb
    // it. The roof stays at 1.58.
    glm::dvec3 bodyHalfExtentsMetres{0.92, 0.64, 2.45};
    double bodyCentreHeightMetres = 0.94;
    // Where the centre of mass sits above that origin. Low, because a car's is: put it at the box's
    // centre and the thing rolls over on every lane change.
    double centreOfMassHeightMetres = 0.55;

    double wheelbaseMetres = 2.95;
    double trackMetres = 1.58;
    double wheelRadiusMetres = 0.36;

    // How far the wheel hangs below its anchor at rest, and how much further it may travel.
    double suspensionRestMetres = 0.28;
    double suspensionTravelMetres = 0.16;

    // Per wheel. The default carries a quarter of 1400 kg at about a third of the travel, which is
    // an ordinary hatchback's static deflection.
    double suspensionStiffnessNewtonsPerMetre = 45000.0;
    double suspensionDampingNewtonSecondsPerMetre = 4500.0;

    // Lateral force per newton of load per radian of slip, and the longitudinal equivalent per unit
    // of slip. Road-tyre figures, and they set how quickly a slide develops rather than how much
    // grip there is — the friction circle below sets that.
    double corneringStiffnessPerNewton = 9.0;
    double longitudinalStiffnessPerNewton = 12.0;
    // The circle's radius, as a coefficient. Multiplied by whatever the surface under each wheel is
    // worth, so a car pushed onto grass loses grip because the road says so.
    double frictionCoefficient = 1.05;

    // What the driver has to work with. A traffic car is not being raced: the drive force is enough
    // to rejoin a lane and the brake is enough to stop. Sized against the mass above — 2.9 m/s^2 of
    // acceleration and 6.9 of braking, which is an ordinary car driven ordinarily.
    double maximumDriveForceNewtons = 5000.0;
    double maximumBrakeForceNewtons = 12000.0;
    double maximumSteerRadians = 0.52;

    // Drag area and rolling resistance, so a car that is left alone slows down and stops rather than
    // coasting for ever.
    double dragAreaSquareMetres = 0.63;
    double rollingResistanceCoefficient = 0.014;
};

export struct SimpleVehicleState
{
    RigidBodyState body;

    // How far each spring is compressed, metres, and what it is carrying, newtons. Kept for the
    // caller to read rather than for the model to remember: every force below is computed from the
    // ray and the velocity, so a state that was reset mid-drive would produce one wrong tick and
    // then be right again.
    std::array<double, simpleWheelCount> compressionMetres{};
    std::array<double, simpleWheelCount> loadNewtons{};
    std::array<double, simpleWheelCount> slipMetresPerSecond{};

    std::uint32_t groundedWheels = 0;
};

export struct SimpleVehicleInput
{
    // -1 to 1. Positive steers left, which is the sense of the +x axis in this engine's frame.
    double steer = 0.0;
    // 0 to 1 each.
    double throttle = 0.0;
    double brake = 0.0;
    // Drive backwards: the drive force turned round, the brake still arresting whatever the wheel
    // is doing. A patrol car backing off something it is stuck against (docs/police-pursuit-brief.md).
    bool reverse = false;
};

// The body's mass, centre of mass and inertia, from the box and the mass. A uniform solid box, which
// puts a 1400 kg hatchback at about 2500 kg·m² in yaw — a real one is nearer 2100, and the
// difference is worth exactly nothing to a car that is being pushed sideways by a Golf.
export void applySimpleVehicleMass(const SimpleVehicleSetup& setup, SimpleVehicleState& state);

// Stand one up at a pose. `positionMetres` is the **body origin** on the road, not the centre of
// mass, so a caller can place one where a lane says a car goes.
export void placeSimpleVehicle(const SimpleVehicleSetup& setup, SimpleVehicleState& state,
                               const glm::dvec3& positionMetres, const glm::dquat& orientation,
                               const glm::dvec3& velocityMetresPerSecond,
                               const glm::dvec3& angularVelocityRadiansPerSecond);

// Where the body origin is, given the state. The inverse of the placement above, for a caller that
// has to draw the thing or ask a lane where it is.
export [[nodiscard]] glm::dvec3 simpleVehicleOrigin(const SimpleVehicleSetup& setup, const SimpleVehicleState& state);

// The bodywork's box, for `collideBody` and for the obstacle the rest of the traffic sees.
export [[nodiscard]] CollisionBox simpleVehicleBox(const SimpleVehicleSetup& setup);

// The four suspension rays, appended to the two arrays in wheel order.
//
// **Appended rather than assigned, because these are meant to be batched.** Every disturbed car's
// rays go into one `PhysicsWorld::castRays` call, on the same principle the contact patch already
// keeps: asking one at a time walks the same broadphase nodes once per wheel.
export void simpleVehicleWheelRays(const SimpleVehicleSetup& setup, const SimpleVehicleState& state,
                                   std::vector<glm::dvec3>& origins, std::vector<glm::dvec3>& directions);

// How far each of those rays has to reach.
export [[nodiscard]] double simpleVehicleRayLength(const SimpleVehicleSetup& setup);

// One tick.
//
// `wheelHits` is this car's four hits, in wheel order, out of the batched cast above. `surfaces` is
// the world's own material table, which is what a hit's `surface` index means — hand it the wrong
// table and a car in the gravel grips like tarmac, which is a fault this project has already had
// once at the contact patch.
//
// `contacts` is the bodywork's manifold, already filled by `collideBody` and `collideObstacles`. It
// is resolved onto the velocity **before** the integrator moves the body, for the reason the full
// model does the same: a contact is a velocity constraint, and a contact expressed as a force over a
// tick is either spongy or explosive depending on the timestep.
export void stepSimpleVehicle(const SimpleVehicleSetup& setup, SimpleVehicleState& state,
                              const SimpleVehicleInput& input, std::span<const SurfaceHit> wheelHits,
                              std::span<const SurfaceMaterial> surfaces, ContactManifold& contacts,
                              const ContactMaterial& material, const double deltaTime);

} // namespace raceengine
