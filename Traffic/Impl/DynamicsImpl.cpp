// Simple traffic vehicle bodies. Declarations are in Api/Dynamics.cppm.
//
// A **module implementation unit** — `module raceengine.traffic;` with no `export`.
module;

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

module raceengine.traffic;

// A partition's own `import` is not re-exported, so this unit names it again.
import raceengine.physics;

namespace raceengine
{

namespace
{

constexpr auto gravityMetresPerSecondSquared = 9.80665;
// Sea level, 20 degrees. The same figure the vehicle model's own aero uses; a traffic car's drag is
// not a number anybody will ever measure.
constexpr auto airDensityKilogramsPerCubicMetre = 1.204;

// How far the spring is compressed with the car standing on level ground, metres.
//
// **The anchor height is derived from it and that is the whole point of the function.** Placed at
// `radius + rest`, the wheel just touches the road at full droop and the body then sinks by this
// much under its own weight — so a car placed on a lane at road level rests 76 mm *inside* the road.
// Subtracting it puts the design position on the road, which is the convention every other body in
// this engine keeps and is what lets a car be promoted out of tier one without jumping.
[[nodiscard]] double staticCompression(const SimpleVehicleSetup& setup)
{
    if (setup.suspensionStiffnessNewtonsPerMetre <= 0.0)
    {
        return 0.0;
    }

    const auto perWheel = setup.massKilograms * gravityMetresPerSecondSquared /
                          (static_cast<double>(simpleWheelCount) * setup.suspensionStiffnessNewtonsPerMetre);

    // A spring too soft to hold the car up is an authoring mistake rather than a case to model; the
    // clamp keeps the anchor above the wheel rather than refusing.
    return std::clamp(perWheel, 0.0, setup.suspensionRestMetres);
}

// Where each wheel's spring is anchored, in the body frame. +x is the car's left.
[[nodiscard]] glm::dvec3 wheelAnchor(const SimpleVehicleSetup& setup, const std::size_t wheel)
{
    const auto left = wheel == static_cast<std::size_t>(SimpleWheel::FrontLeft) ||
                      wheel == static_cast<std::size_t>(SimpleWheel::RearLeft);
    const auto front = wheel == static_cast<std::size_t>(SimpleWheel::FrontLeft) ||
                       wheel == static_cast<std::size_t>(SimpleWheel::FrontRight);

    return glm::dvec3((left ? 0.5 : -0.5) * setup.trackMetres,
                      setup.wheelRadiusMetres + setup.suspensionRestMetres - staticCompression(setup),
                      (front ? 0.5 : -0.5) * setup.wheelbaseMetres);
}

[[nodiscard]] glm::dvec3 pointVelocity(const RigidBodyState& body, const glm::dvec3& worldPoint)
{
    return body.linearVelocity + glm::cross(angularVelocity(body), worldPoint - body.position);
}

} // namespace

double simpleVehicleRayLength(const SimpleVehicleSetup& setup)
{
    // Long enough to find the road with the wheel at full droop, which is the anchor's own height
    // plus the whole free length, plus the travel the body may still be lifted by.
    return setup.wheelRadiusMetres + setup.suspensionRestMetres + setup.suspensionTravelMetres;
}

CollisionBox simpleVehicleBox(const SimpleVehicleSetup& setup)
{
    return CollisionBox{.centre = glm::dvec3(0.0, setup.bodyCentreHeightMetres, 0.0),
                        .halfExtents = setup.bodyHalfExtentsMetres};
}

void applySimpleVehicleMass(const SimpleVehicleSetup& setup, SimpleVehicleState& state)
{
    const auto full = 2.0 * setup.bodyHalfExtentsMetres;
    const auto mass = setup.massKilograms;

    state.body.mass = mass;
    state.body.centreOfMass = glm::dvec3(0.0, setup.centreOfMassHeightMetres, 0.0);

    const auto inertia = glm::dmat3(1.0 / 12.0 * mass * (full.y * full.y + full.z * full.z), 0.0, 0.0, 0.0,
                                    1.0 / 12.0 * mass * (full.x * full.x + full.z * full.z), 0.0, 0.0, 0.0,
                                    1.0 / 12.0 * mass * (full.x * full.x + full.y * full.y));

    state.body.inertia = inertia;
    state.body.inverseInertia = glm::inverse(inertia);
}

void placeSimpleVehicle(const SimpleVehicleSetup& setup, SimpleVehicleState& state, const glm::dvec3& positionMetres,
                        const glm::dquat& orientation, const glm::dvec3& velocityMetresPerSecond,
                        const glm::dvec3& angularVelocityRadiansPerSecond)
{
    applySimpleVehicleMass(setup, state);

    state.body.orientation = orientation;
    // `position` is the centre of mass, so the body origin has to be turned before it is offset.
    state.body.position = positionMetres + orientation * state.body.centreOfMass;
    state.body.linearVelocity = velocityMetresPerSecond;
    setAngularVelocity(state.body, angularVelocityRadiansPerSecond);

    state.compressionMetres = {};
    state.loadNewtons = {};
    state.slipMetresPerSecond = {};
    state.groundedWheels = 0;
}

glm::dvec3 simpleVehicleOrigin(const SimpleVehicleSetup&, const SimpleVehicleState& state)
{
    return state.body.position - state.body.orientation * state.body.centreOfMass;
}

void simpleVehicleWheelRays(const SimpleVehicleSetup& setup, const SimpleVehicleState& state,
                            std::vector<glm::dvec3>& origins, std::vector<glm::dvec3>& directions)
{
    // Down the car's own suspension axis rather than down the world's: a car on its side has to find
    // out that its wheels are not touching anything, and a world-down ray from a rolled car finds the
    // road under its roof.
    const auto down = state.body.orientation * glm::dvec3(0.0, -1.0, 0.0);

    for (auto wheel = std::size_t{0}; wheel < simpleWheelCount; wheel++)
    {
        origins.push_back(bodyToWorld(state.body, wheelAnchor(setup, wheel)));
        directions.push_back(down);
    }
}

void stepSimpleVehicle(const SimpleVehicleSetup& setup, SimpleVehicleState& state, const SimpleVehicleInput& input,
                       const std::span<const SurfaceHit> wheelHits, const std::span<const SurfaceMaterial> surfaces,
                       ContactManifold& contacts, const ContactMaterial& material, const double deltaTime)
{
    if (deltaTime <= 0.0 || state.body.mass <= 0.0)
    {
        return;
    }

    auto forces = ForceAccumulator{};

    forces.force += glm::dvec3(0.0, -gravityMetresPerSecondSquared * state.body.mass, 0.0);

    // Body drag, and it is the only aero here. A traffic car's is not a measured number and is not
    // going to become one; what it is for is making a car that has been knocked loose come to rest.
    const auto speed = glm::length(state.body.linearVelocity);
    if (speed > 1e-6)
    {
        forces.force -=
            0.5 * airDensityKilogramsPerCubicMetre * setup.dragAreaSquareMetres * speed * state.body.linearVelocity;
    }

    const auto steerAngle = std::clamp(input.steer, -1.0, 1.0) * setup.maximumSteerRadians;
    const auto driveForce =
        std::clamp(input.throttle, 0.0, 1.0) * setup.maximumDriveForceNewtons * (input.reverse ? -1.0 : 1.0);
    const auto brakeForce = std::clamp(input.brake, 0.0, 1.0) * setup.maximumBrakeForceNewtons;

    state.groundedWheels = 0;

    for (auto wheel = std::size_t{0}; wheel < simpleWheelCount; wheel++)
    {
        state.compressionMetres[wheel] = 0.0;
        state.loadNewtons[wheel] = 0.0;
        state.slipMetresPerSecond[wheel] = 0.0;

        if (wheel >= wheelHits.size() || !wheelHits[wheel].hit)
        {
            continue;
        }

        const auto& hit = wheelHits[wheel];

        // How far the spring is compressed: the wheel would hang `suspensionRestMetres` below its
        // anchor with nothing under it, and the road is `distance - radius` below it instead.
        const auto hang = hit.distance - setup.wheelRadiusMetres;
        const auto compression = setup.suspensionRestMetres - hang;
        if (compression <= 0.0)
        {
            continue;
        }

        state.groundedWheels++;
        state.compressionMetres[wheel] = compression;

        const auto anchor = bodyToWorld(state.body, wheelAnchor(setup, wheel));
        const auto contact = anchor + hit.distance * (state.body.orientation * glm::dvec3(0.0, -1.0, 0.0));
        const auto velocity = pointVelocity(state.body, contact);

        // The surface normal, and the spring pushing along it. Closing speed rather than a
        // difference of two compressions, so the damper reads a velocity the integrator agrees with
        // rather than one reconstructed from history.
        const auto normal = glm::length(hit.normal) > 1e-9 ? glm::normalize(hit.normal) : glm::dvec3(0.0, 1.0, 0.0);
        const auto closing = -glm::dot(velocity, normal);

        const auto load = std::max(0.0, setup.suspensionStiffnessNewtonsPerMetre * compression +
                                            setup.suspensionDampingNewtonSecondsPerMetre * closing);

        if (load <= 0.0)
        {
            continue;
        }

        state.loadNewtons[wheel] = load;
        forces.addForceAtPoint(load * normal, contact, state.body.position);

        // The damper handed to the integrator as a coefficient as well as a force, so no stiffness
        // of it can overshoot. It is the linear part only — the moment arm's contribution is left
        // explicit — and at a road car's rate against a 360 Hz tick the explicit part is nowhere
        // near the limit anyway: c·dt/m is 0.03.
        forces.linearDamping += setup.suspensionDampingNewtonSecondsPerMetre * glm::outerProduct(normal, normal);

        // --- the tyre --------------------------------------------------------------------------

        const auto front = wheel == static_cast<std::size_t>(SimpleWheel::FrontLeft) ||
                           wheel == static_cast<std::size_t>(SimpleWheel::FrontRight);
        const auto steer = front ? steerAngle : 0.0;

        const auto heading =
            state.body.orientation * glm::angleAxis(steer, glm::dvec3(0.0, 1.0, 0.0)) * glm::dvec3(0.0, 0.0, 1.0);

        // Both axes in the contact plane, so a cambered road does not leak grip into the normal.
        auto forward = heading - glm::dot(heading, normal) * normal;
        if (glm::length(forward) < 1e-6)
        {
            continue;
        }

        forward = glm::normalize(forward);
        // `cross(up, forward)` is the car's left, which is the sense every lateral quantity in this
        // engine is quoted in.
        const auto lateral = glm::cross(normal, forward);

        const auto alongSpeed = glm::dot(velocity, forward);
        const auto acrossSpeed = glm::dot(velocity, lateral);

        state.slipMetresPerSecond[wheel] = acrossSpeed;

        const auto grip = hit.surface < surfaces.size() ? surfaces[hit.surface].gripMultiplier : 1.0;
        const auto limit = setup.frictionCoefficient * grip * load;

        // A slip *velocity* rather than a slip angle, softened by a floor on the reference speed.
        // The two agree above walking pace and only this one is finite at a standstill, which is
        // where a parked traffic car spends most of its life.
        const auto reference = std::max(std::abs(alongSpeed), 2.0);

        auto across = -setup.corneringStiffnessPerNewton * load * acrossSpeed / reference;
        auto along = driveForce * 0.5;

        if (brakeForce > 0.0)
        {
            // Enough to stop the wheel rather than a signed constant, so a car on the brakes comes to
            // rest instead of shuffling back and forth across zero at 360 Hz. The cap is the pedal.
            const auto arrest = -alongSpeed * state.body.mass / (4.0 * deltaTime);

            along += std::clamp(arrest, -brakeForce, brakeForce);
        }

        // Rolling resistance, which is what brings a coasting car to a stop.
        along -= setup.rollingResistanceCoefficient * load * std::tanh(alongSpeed / 0.5);

        // The friction circle. Scaled as a pair rather than clamped one axis at a time, for the
        // reason the contact solver's own friction is: an axis-at-a-time clamp lets a wheel sliding
        // at 45 degrees take sqrt(2) times the grip of one sliding straight.
        const auto magnitude = std::sqrt(along * along + across * across);
        if (magnitude > limit && magnitude > 1e-9)
        {
            const auto scale = limit / magnitude;

            along *= scale;
            across *= scale;
        }

        forces.addForceAtPoint(along * forward + across * lateral, contact, state.body.position);
    }

    // Bodywork contact resolved onto the velocity, before the integrator moves anything.
    resolveContacts(state.body, contacts, material, deltaTime);

    integrate(state.body, forces, deltaTime);

    // A quaternion that rounding has pulled off the unit sphere turns into a shear on the next
    // matrix cast. The integrator renormalises its own; this is the belt for a state a caller may
    // have written by hand.
    state.body.orientation = glm::normalize(state.body.orientation);
}

} // namespace raceengine
