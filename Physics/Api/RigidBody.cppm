module;

#include <cmath>
#include <expected>
#include <span>
#include <string>
#include <type_traits>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module raceengine.physics:RigidBody;

namespace raceengine
{

// The chassis integrator, stated away from any device or scene for the reason PhysicalCamera is:
// what a body does over a tick is arithmetic, and arithmetic can be read and tested without a
// window. Nothing in this partition knows about Jolt, the renderer, or the scene graph.
//
// **SI throughout** — metres, kilograms, seconds, radians, newtons — and deliberately not the
// engine's world units. Every coefficient the vehicle model will carry is quoted in SI in the
// literature it comes from, and a tire relaxation length that is secretly in decimetres is
// indistinguishable from a badly chosen one. The conversion belongs at the single point where a
// body's transform is written into a scene node, and this milestone has no such point.
//
// **Double, not float**, which is a decision worth stating once: an undamped orbit has to hold its
// energy over ten thousand ticks, an open world puts a car kilometres from the origin where float
// has lost the millimetre, and the state below is serialised and round-tripped, so its width is
// free to choose now and expensive to change once telemetry and save files are written against it.

// One item on the mass ledger: a chassis, a fuel load, an occupant. Kept as a list rather than
// folded into a constant because fuel burns off and occupants get out — mass, centre of gravity
// and inertia are *derived* every time that list changes rather than authored once, which is the
// seam a later fuel model needs and the one most easily missed.
export struct MassComponent
{
    double mass = 0.0;
    // Body frame, relative to the model origin — not to the centre of mass, which is what this
    // list is being used to find.
    glm::dvec3 centre{0.0};
    // About this component's own centre, in body axes. Zero is a legitimate answer for a point
    // mass such as a fuel load small enough that only its position matters.
    glm::dmat3 inertia{0.0};
};

export struct MassProperties
{
    double mass = 1.0;
    // Body frame, relative to the model origin.
    glm::dvec3 centreOfMass{0.0};
    // About the centre of mass, in body axes.
    glm::dmat3 inertia{1.0};
    glm::dmat3 inverseInertia{1.0};
};

// The whole of a body's dynamic state, and nothing else: no pointers, no handles, no cached
// derived quantities that a restore could contradict. Trivially copyable and standard layout, so
// save and restore are a memcpy — which the validation harness needs now and rollback needs later.
export struct RigidBodyState
{
    // World frame. This is the centre of mass, not the model origin; `centreOfMass` below is what
    // reconstructs the render transform from it.
    glm::dvec3 position{0.0};
    glm::dquat orientation{1.0, 0.0, 0.0, 0.0};
    glm::dvec3 linearVelocity{0.0};

    // Angular *momentum*, world frame, kg·m²/s — and this is the one storage choice here that is
    // not the obvious one, so: with no torque acting, momentum is constant and angular velocity is
    // not. Integrating momentum and deriving omega from it each tick therefore conserves momentum
    // exactly, and reproduces the free rotation of a body whose principal inertias differ — the
    // intermediate-axis flip — with no explicit omega x (I omega) term to go unstable at large
    // omega. Storing omega instead makes that term the integrator's problem rather than the
    // representation's. `angularVelocity` reads it out for anything that wants it.
    glm::dvec3 angularMomentum{0.0};

    // Recomputable at runtime rather than fixed at construction, which is why they live in the
    // state and not beside it: burning fuel changes all four.
    double mass = 1.0;
    glm::dvec3 centreOfMass{0.0};
    glm::dmat3 inertia{1.0};
    glm::dmat3 inverseInertia{1.0};
};

static_assert(std::is_trivially_copyable_v<RigidBodyState>,
              "the validation harness and any later rollback save this by copying its bytes");
static_assert(std::is_standard_layout_v<RigidBodyState>, "and read them back on the other side of a file");

// What every contributing module writes into, cleared at the top of the tick and applied at the
// bottom. No module writes a velocity or a position: that separation is what makes suspension,
// tire, aero and driveline independently testable, and what makes substepping a change to the
// loop rather than to any of them.
export struct ForceAccumulator
{
    // World frame. Torque is about the centre of mass.
    glm::dvec3 force{0.0};
    glm::dvec3 torque{0.0};

    // The velocity-proportional part of what the contributors above are applying — dF/dv in N·s/m
    // and dTau/dOmega in N·m·s — handed over *separately* from `force` rather than folded into it
    // as -C·v.
    //
    // This is the single most important line in the file. A damper contributed explicitly is a
    // force computed from the velocity the body had at the start of the tick, and once C·dt/m
    // exceeds two that overshoots by more than it corrects: the body oscillates, then diverges. It
    // is the classic way a hand-written vehicle integrator dies, and it dies at exactly the damping
    // rates a real car runs. Given the coefficient instead of the product, `integrate` solves for
    // the velocity that is consistent with it and is stable at any stiffness.
    glm::dmat3 linearDamping{0.0};
    glm::dmat3 angularDamping{0.0};

    void clear()
    {
        *this = ForceAccumulator{};
    }

    // The one way a contributor applies a force somewhere other than the centre of mass, and the
    // reason the suspension does not need to model jacking explicitly: a spring force at the damper
    // mount and a tire force at the contact patch produce their own moments because they are
    // applied where they act.
    void addForceAtPoint(const glm::dvec3& newtons, const glm::dvec3& worldPoint, const glm::dvec3& worldCentreOfMass)
    {
        force += newtons;
        torque += glm::cross(worldPoint - worldCentreOfMass, newtons);
    }
};

// glm's dmat3(0.0) is the zero matrix and dmat3(1.0) the identity, both by the diagonal
// constructor. Named because `== glm::dmat3(0.0)` reads as a comparison against a scalar. Not
// exported and not in a private fragment: a partition may not have one.
const auto zeroMatrix = glm::dmat3(0.0);
const auto identityMatrix = glm::dmat3(1.0);

export glm::dmat3 worldInertia(const RigidBodyState& state);

export glm::dmat3 worldInverseInertia(const RigidBodyState& state);

export glm::dvec3 angularVelocity(const RigidBodyState& state);

export void setAngularVelocity(RigidBodyState& state, const glm::dvec3& radiansPerSecond);

export double kineticEnergy(const RigidBodyState& state);

export glm::dvec3 bodyToWorld(const RigidBodyState& state, const glm::dvec3& bodyPoint);

export std::expected<MassProperties, std::string> computeMassProperties(std::span<const MassComponent> components);

export void applyMassProperties(RigidBodyState& state, const MassProperties& properties);

export void integrate(RigidBodyState& state, const ForceAccumulator& forces, const double deltaTime);

} // namespace raceengine
