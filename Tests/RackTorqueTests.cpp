#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine;

using raceengine::golfGtiMk7;
using raceengine::pinionRadius;
using raceengine::solveCorner;
using raceengine::SteeredCorner;
using raceengine::steeredCornerLimit;
using raceengine::SteeringRack;
using raceengine::steeringRackTorque;

namespace
{

// A corner with its kingpin straight up and its steering arm square to it, so every number this
// produces can be worked out on paper. Nothing in the real car looks like this and that is the
// point: a case whose answer is arithmetic is the only one that can catch a sign.
SteeredCorner squareCorner(const double trail = 0.04, const double armLength = 0.15, const double scrub = 0.0,
                           const double kingpinLean = 0.0)
{
    return SteeredCorner{.lowerBallJoint = glm::dvec3(0.75, 0.10, 0.0),
                         // Leaning inboard by `kingpinLean` over its half metre of height, which is
                         // kingpin inclination. Zero makes the axis exactly vertical, which is the
                         // case whose answers are arithmetic.
                         .upperBallJoint = glm::dvec3(0.75 - kingpinLean, 0.60, 0.0),
                         // Behind the axis by `armLength`, which is the lever the rack pulls on.
                         .steeringArm = glm::dvec3(0.75, 0.10, -armLength),
                         .rackOuter = glm::dvec3(0.30, 0.10, -armLength),
                         // Ahead of the axis by `trail`: mechanical trail, as geometry rather than
                         // as a number written into the tyre.
                         // Ahead of the axis by `trail` and outboard of it by `scrub`. The two do
                         // different jobs: trail is the lever a side force pulls on, scrub is the
                         // lever a vertical load pulls on, and neither substitutes for the other.
                         .contactPatch = glm::dvec3(0.75 + scrub, 0.0, trail),
                         .patchNormal = glm::dvec3(0.0, 1.0, 0.0),
                         .tyreForce = glm::dvec3(0.0),
                         .aligningMoment = 0.0};
}

} // namespace

TEST_CASE("the rack reports newton metres at the rim and the pinion is the whole ratio", "[input][ffb][rack]")
{
    const auto rack = SteeringRack{};

    // Rack and pinion: there is no second reduction between the pinion and the rim, so this radius
    // is the entire steering ratio and the entire unit conversion.
    const auto radius = pinionRadius(rack);
    REQUIRE(radius > 0.0);
    REQUIRE(radius ==
            Catch::Approx(2.0 * rack.travelPerInput / (rack.lockToLockDegrees * 3.14159265358979323846 / 180.0)));

    auto corner = squareCorner();
    corner.tyreForce = glm::dvec3(0.0, 0.0, 0.0);

    const auto corners = std::array<SteeredCorner, 1>{corner};
    const auto answer = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 0.0);

    REQUIRE(answer.finite);
    REQUIRE(answer.steeringTorque == Catch::Approx(answer.rackForce * radius));
}

TEST_CASE("a tyre's side force reaches the rim through the geometry and nothing else", "[input][ffb][rack]")
{
    const auto rack = SteeringRack{};

    SECTION("mechanical trail is the patch's offset, not a term")
    {
        // A side force acting a distance ahead of the kingpin axis is a moment about it of exactly
        // force times that distance. Doubling the trail doubles the moment, and there is nowhere in
        // the model this could come from other than where the patch is.
        const auto side = 4000.0;

        auto near = squareCorner(0.02);
        auto far = squareCorner(0.04);
        near.tyreForce = glm::dvec3(side, 0.0, 0.0);
        far.tyreForce = glm::dvec3(side, 0.0, 0.0);

        const auto one = std::array<SteeredCorner, 1>{near};
        const auto two = std::array<SteeredCorner, 1>{far};

        const auto shallow = steeringRackTorque(rack, std::span<const SteeredCorner>(one), 0.0);
        const auto deep = steeringRackTorque(rack, std::span<const SteeredCorner>(two), 0.0);

        REQUIRE(shallow.kingpinTorque[0] == Catch::Approx(side * 0.02).epsilon(1e-9));
        REQUIRE(deep.kingpinTorque[0] == Catch::Approx(side * 0.04).epsilon(1e-9));
        REQUIRE(deep.kingpinTorque[0] == Catch::Approx(2.0 * shallow.kingpinTorque[0]));
    }

    SECTION("and the self-aligning moment arrives about the patch normal")
    {
        auto upright = squareCorner();
        upright.aligningMoment = 90.0;

        const auto corners = std::array<SteeredCorner, 1>{upright};
        const auto answer = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 0.0);

        // The kingpin is vertical here and so is the patch normal, so all of it lands on the axis.
        REQUIRE(answer.kingpinTorque[0] == Catch::Approx(90.0));

        // Laid on its side, none of it does — which is what stops a car on steep camber being
        // steered by an aligning moment that is no longer pointing at the steering.
        auto sideways = upright;
        sideways.patchNormal = glm::dvec3(1.0, 0.0, 0.0);

        const auto tipped = std::array<SteeredCorner, 1>{sideways};
        REQUIRE(steeringRackTorque(rack, std::span<const SteeredCorner>(tipped), 0.0).kingpinTorque[0] ==
                Catch::Approx(0.0).margin(1e-9));
    }

    SECTION("and a vertical load through an offset axis is steering weight on its own")
    {
        // What a parked car's steering feels, and the reason `tyreForce` is the whole resultant
        // rather than the lateral component: no side force here at all.
        //
        // It takes an *inclined* kingpin, and that is worth pinning because it is the whole
        // mechanism by which a car's steering self-centres standing still and goes heavy at parking
        // speed. A vertical force has no moment whatever about a vertical axis, however far the
        // patch is offset from it — the cross product of a vertical lever arm component with a
        // vertical force is perpendicular to both. Lean the axis and it appears.
        const auto load = 3500.0;

        auto upright = squareCorner(0.04, 0.15, 0.02, 0.0);
        upright.tyreForce = glm::dvec3(0.0, load, 0.0);

        const auto vertical = std::array<SteeredCorner, 1>{upright};
        REQUIRE(steeringRackTorque(rack, std::span<const SteeredCorner>(vertical), 0.0).kingpinTorque[0] ==
                Catch::Approx(0.0).margin(1e-9));

        auto leaning = squareCorner(0.04, 0.15, 0.02, 0.06);
        leaning.tyreForce = glm::dvec3(0.0, load, 0.0);

        const auto inclined = std::array<SteeredCorner, 1>{leaning};
        const auto weighted = steeringRackTorque(rack, std::span<const SteeredCorner>(inclined), 0.0);

        REQUIRE(std::abs(weighted.kingpinTorque[0]) > 1.0);

        // And it is proportional to the load, which is what makes a kerb strike reach the hands.
        auto heavier = leaning;
        heavier.tyreForce = glm::dvec3(0.0, 2.0 * load, 0.0);

        const auto loaded = std::array<SteeredCorner, 1>{heavier};
        REQUIRE(steeringRackTorque(rack, std::span<const SteeredCorner>(loaded), 0.0).kingpinTorque[0] ==
                Catch::Approx(2.0 * weighted.kingpinTorque[0]).epsilon(1e-9));
    }
}

TEST_CASE("the rack's own resistances oppose it and can never drive it", "[input][ffb][rack]")
{
    const auto rack = SteeringRack{};
    const auto corners = std::array<SteeredCorner, 1>{squareCorner()};

    SECTION("both take the sign of the motion")
    {
        for (const auto velocity : {-0.4, -0.05, 0.05, 0.4})
        {
            const auto answer = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), velocity);

            REQUIRE(answer.frictionForce * velocity < 0.0);
            REQUIRE(answer.dampingForce * velocity < 0.0);
            // Coulomb friction is bounded by its own coefficient however fast the rack moves.
            REQUIRE(std::abs(answer.frictionForce) <= rack.friction);
        }
    }

    SECTION("and the friction is regularised through zero rather than switched")
    {
        // `-F * sign(v)` flips its whole magnitude between two consecutive ticks at a standstill.
        // On a base running at 500 Hz that is an audible buzz, and on a trace it is a square wave
        // nobody can read past. What says this is regularised is that a rack barely moving feels
        // barely any of it.
        const auto crawling = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 1e-4);
        const auto moving = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 1.0);

        REQUIRE(std::abs(crawling.frictionForce) < 0.05 * rack.friction);
        REQUIRE(std::abs(moving.frictionForce) == Catch::Approx(rack.friction).epsilon(0.01));

        // And it is continuous across the sign change, which is the property being bought.
        const auto below = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), -1e-6);
        const auto above = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 1e-6);
        // 0.024 N across the crossing, against a 120 N coefficient: two hundredths of a percent of
        // the force the switched form would have flipped through.
        REQUIRE(std::abs(above.frictionForce - below.frictionForce) < 0.001 * rack.friction);
    }
}

TEST_CASE("a stage-one refusal is a refusal rather than a number", "[input][ffb][rack]")
{
    const auto rack = SteeringRack{};
    const auto nan = std::numeric_limits<double>::quiet_NaN();

    SECTION("a rack velocity that is not a number")
    {
        const auto corners = std::array<SteeredCorner, 1>{squareCorner()};
        REQUIRE_FALSE(steeringRackTorque(rack, std::span<const SteeredCorner>(corners), nan).finite);
    }

    SECTION("and any corner that is not a number, in any of its fields")
    {
        auto broken = squareCorner();
        broken.tyreForce = glm::dvec3(nan, 0.0, 0.0);

        const auto corners = std::array<SteeredCorner, 1>{broken};
        REQUIRE_FALSE(steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 0.0).finite);

        auto alsoBroken = squareCorner();
        alsoBroken.aligningMoment = nan;

        const auto second = std::array<SteeredCorner, 1>{alsoBroken};
        REQUIRE_FALSE(steeringRackTorque(rack, std::span<const SteeredCorner>(second), 0.0).finite);
    }
}

namespace
{

// Where this corner's kingpin axis pierces the road, and how far the contact patch trails it.
// Positive is the patch behind the axis, which is the sense that makes a tyre self-centre.
//
// Read off the *solved* geometry rather than authored anywhere, which is what makes it an
// independent statement: move a hardpoint and this moves, and nothing in `steeringRackTorque` is
// consulted to work it out.
[[nodiscard]] double mechanicalTrailOf(const raceengine::SuspensionState& solved)
{
    const auto span = solved.upperBallJoint - solved.lowerBallJoint;
    const auto intercept = solved.lowerBallJoint + (solved.contactPatch.y - solved.lowerBallJoint.y) / span.y * span;

    return intercept.z - solved.contactPatch.z;
}

// The wheel's heading — its rotation about world +y — read off the upright's own forward direction.
//
// **Not the toe angle**, and the difference is load bearing. `SuspensionState::toe` is measured off
// a spin axis that points *away from the car*, so it is a mirrored quantity and its two sides carry
// opposite signs for the same physical turn. Anything summing signed contributions from both corners
// would, fed toe, be summing a left turn and a right one. Reading the nose is exactly the correction
// `rackTravelForSteer` had to make when the frame convention was settled.
[[nodiscard]] double headingOf(const raceengine::SuspensionState& solved)
{
    const auto nose = solved.uprightOrientation * glm::dvec3(0.0, 0.0, 1.0);

    return std::atan2(nose.x, nose.z);
}

// The rotation the upright turns through about its own kingpin axis, between two solves.
//
// This is the exact quantity the virtual-work identity needs, and reading the *heading* instead is
// only correct to first order: the kingpin leans, so a rotation about it tilts the wheel as well as
// aiming it, and the nose has a caster-sized pitch that the flat `atan2` does not account for.
// Measured on this car that is worth 0.22% at 20 mm of rack — small, real, and not something to
// absorb into a tolerance when the exact statement is four lines.
//
// Independent of `steeringRackTorque` by construction: that derives its Jacobian from the tie rod as
// a two-force member, and this differentiates the linkage solve, which reaches the tie rod as a
// distance constraint. They have to agree, and them agreeing is worth having.
[[nodiscard]] double kingpinRotationBetween(const raceengine::SuspensionState& from,
                                            const raceengine::SuspensionState& to, const glm::dvec3& kingpin)
{
    auto relative = to.uprightOrientation * glm::conjugate(from.uprightOrientation);
    if (relative.w < 0.0)
    {
        relative = -relative;
    }

    // Small-angle exact enough at a millimetre of rack: the rotation vector is twice the quaternion's
    // vector part to well past the precision anything here is compared at.
    const auto rotation = 2.0 * glm::dvec3(relative.x, relative.y, relative.z);

    return glm::dot(rotation, kingpin);
}

} // namespace

TEST_CASE("criterion 10: the rack reports the trail its own hardpoints imply", "[input][ffb][rack][published]")
{
    // **Criterion 10, rewritten as an identity rather than as a range.**
    //
    // What it used to assert was `2.0 < atTheRim < 50.0` against a measured 19.9 — a twenty-five
    // times window, which is a check that passes for any plausible scale factor anywhere in the
    // chain. That is not a hypothetical failure mode: a hardware-sized assist of 0.22 to 0.55 sat
    // between this derivation and the wheel for the whole of milestone 2 and this test could not
    // have seen it, because 0.22 x 19.9 is still comfortably inside the window.
    //
    // So the magnitude check is replaced by a relationship the geometry already knows. A lateral
    // force `Fy` at a contact patch that trails the kingpin's ground intercept by `t` makes a moment
    // `-t.Fy.k_y` about that axis, and virtual work turns each corner's moment into rim torque
    // through the rate the linkage turns about its own kingpin:
    //
    //     T_rim  =  -r_pinion . SUM_i ( t_i . Fy_i . k_y,i . ddelta_i/dx )
    //
    // with `delta` the rotation about the kingpin and `x` the rack. Every term on the right is read
    // off the solved linkage and none of it is read off the function under test. A scale factor
    // cannot satisfy it, a doubled corner count cannot satisfy it, and a wrong pinion radius cannot
    // satisfy it — which are the three ways the old window was blind.
    const auto car = golfGtiMk7();
    REQUIRE(car.has_value());

    // **This car's own rack, not the model's default.** The old test measured through a
    // `SteeringRack{}` whose 0.055 m of travel per unit is not what this car has (0.0700), so its
    // recorded 19.9 N·m was through a pinion the Golf does not turn. Through the right one the same
    // axle reads 30.5.
    auto rack = SteeringRack{};
    rack.travelPerInput = car->rackTravelPerInput;
    rack.lockToLockDegrees = 756.0;

    const auto radius = pinionRadius(rack);

    // A front axle at the limit: this car's front pair carries about 8 kN and a semislick makes
    // rather more than its own load in side force at peak, so 4 kN a corner is the honest figure for
    // one wheel of a Golf on the edge of grip.
    constexpr auto sideForce = 4000.0;
    constexpr auto verticalLoad = 4000.0;
    // Pneumatic trail at peak slip, folded into Mz where it belongs. About 25 mm on a road tyre, and
    // it falls to nothing past the peak, which is the lightening a driver steers by.
    constexpr auto pneumaticTrail = 0.025;

    // Off centre as well as centred. Centred, the two corners are mirror images and the identity
    // could be satisfied by a formula that had lost one of them; twenty millimetres of rack puts
    // Ackermann between them, so the two trails and the two steering rates genuinely differ and the
    // sum has to be a sum.
    const auto rackTravel = GENERATE(0.0, 0.020, -0.020);
    CAPTURE(rackTravel);

    auto corners = std::array<SteeredCorner, 2>{};
    auto predicted = 0.0;
    auto trails = std::array<double, 2>{};

    for (auto index = std::size_t{0}; index < 2; index++)
    {
        const auto& hardpoints = car->corners[index].hardpoints;

        const auto solved = solveCorner(hardpoints, 0.0, rackTravel);
        REQUIRE(solved.has_value());

        // The steering rate, by central difference. A millimetre either side: far enough out of the
        // solve's own noise, far short of any distance Ackermann moves in.
        constexpr auto step = 0.001;
        const auto ahead = solveCorner(hardpoints, 0.0, rackTravel + step);
        const auto behind = solveCorner(hardpoints, 0.0, rackTravel - step);
        REQUIRE(ahead.has_value());
        REQUIRE(behind.has_value());

        const auto kingpin = glm::normalize(solved->upperBallJoint - solved->lowerBallJoint);
        const auto turnPerTravel = kingpinRotationBetween(behind.value(), ahead.value(), kingpin) / (2.0 * step);

        trails[index] = mechanicalTrailOf(solved.value());
        predicted += -radius * (trails[index] + pneumaticTrail) * sideForce * kingpin.y * turnPerTravel;

        // The readable diagnostic beside the exact one: how many degrees of rim this corner takes
        // per degree of road wheel. Measured 13.80:1 centred, which is what a hatchback runs.
        const auto headingPerTravel = (headingOf(ahead.value()) - headingOf(behind.value())) / (2.0 * step);
        const auto ratio = 1.0 / (radius * std::abs(headingPerTravel));
        CAPTURE(index, ratio);
        REQUIRE(ratio > 8.0);
        REQUIRE(ratio < 22.0);

        corners[index] = SteeredCorner{.lowerBallJoint = solved->lowerBallJoint,
                                       .upperBallJoint = solved->upperBallJoint,
                                       .steeringArm = solved->steeringArm,
                                       .rackOuter = hardpoints.steeringRackOuter + glm::dvec3(rackTravel, 0.0, 0.0),
                                       .contactPatch = solved->contactPatch,
                                       .patchNormal = glm::dvec3(0.0, 1.0, 0.0),
                                       .tyreForce = glm::dvec3(sideForce, 0.0, 0.0),
                                       .aligningMoment = -sideForce * pneumaticTrail};
    }

    // **Side force alone, which is what the identity is about.** The vertical load is added back in
    // its own section below, because it reaches the kingpin through the *scrub radius* rather than
    // through the trail and is therefore a different statement, not a term to fold in and blur.
    const auto answer = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 0.0);
    REQUIRE(answer.finite);

    CAPTURE(answer.kingpinTorque[0], answer.kingpinTorque[1], answer.tyreForce, answer.steeringTorque, predicted,
            trails[0], trails[1]);

    // **The identity.** One part in ten thousand, and that is headroom rather than tolerance: the
    // two sides agree to better than one part in fifty thousand at every rack position here, which
    // is the central difference's own truncation and nothing else. Against the twenty-five times
    // window this replaces, that is four orders of magnitude of discrimination.
    REQUIRE(answer.steeringTorque == Catch::Approx(predicted).epsilon(0.0001));

    // And the absolute-units half of the criterion, which is now a statement about the *geometry*
    // rather than about the torque. A road car's mechanical trail is a couple of centimetres, set by
    // caster and the kingpin's ground intercept, and Ackermann spreads the two corners apart as the
    // rack moves off centre — 12.5 mm on the inner wheel against 42.0 on the outer at 20 mm of rack,
    // against 27.6 mm on both when it is centred. A plausible-looking scale factor cannot reach any
    // of those numbers, which is the whole difference between this and the window it replaces.
    for (const auto trail : trails)
    {
        REQUIRE(trail > 0.005);
        REQUIRE(trail < 0.055);
    }

    SECTION("and a vertical load reaches the rack through the scrub radius, cancelling side to side")
    {
        // The other lever, and the one that a wheel meets a kerb through. Two equally loaded front
        // wheels have mirror-image scrub radii, so at centred steering their contributions cancel to
        // the bit — which is why a parked car's steering weight is a *tyre* fact rather than a
        // load fact, and why what a driver feels over a bump is the load *difference*.
        for (auto& corner : corners)
        {
            corner.tyreForce = glm::dvec3(0.0, verticalLoad, 0.0);
            corner.aligningMoment = 0.0;
        }

        const auto vertical = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 0.0);
        REQUIRE(vertical.finite);
        CAPTURE(vertical.steeringTorque);

        if (rackTravel == 0.0)
        {
            REQUIRE(vertical.steeringTorque == Catch::Approx(0.0).margin(1e-12));
        }
        else
        {
            // Off centre the mirror is broken and it does not cancel — but it stays small beside the
            // trail term, which is what says the steering is loaded by the road rather than by the
            // car's weight.
            REQUIRE(std::abs(vertical.steeringTorque) < 0.25 * std::abs(answer.steeringTorque));
        }

        // One wheel unloaded and the other not: the cancellation goes and what is left is the
        // asymmetry, in the direction the loaded wheel's scrub pulls.
        corners[1].tyreForce = glm::dvec3(0.0);

        const auto uneven = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 0.0);
        REQUIRE(uneven.finite);
        REQUIRE(std::abs(uneven.steeringTorque) > 1.0);
    }

    SECTION("the whole axle at the limit, in absolute newton metres")
    {
        for (auto& corner : corners)
        {
            corner.tyreForce = glm::dvec3(sideForce, verticalLoad, 0.0);
        }

        const auto whole = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 0.0);
        REQUIRE(whole.finite);

        const auto atTheRim = std::abs(whole.steeringTorque);
        CAPTURE(atTheRim);

        // Measured 30.5 N·m centred, unassisted, through this car's own 10.6 mm pinion. **Note this
        // is not the 19.9 recorded through milestone 2**, which was taken through a
        // `SteeringRack{}`'s 8.3 mm pinion — a rack this car does not have.
        //
        // Enough that a driver's arms know about it, far short of the hundreds a kingpin moment
        // reads before the pinion divides it down. This is still a range and it is deliberately the
        // *weaker* of the two checks here: what makes the criterion hold is the identity above.
        REQUIRE(atTheRim > 20.0);
        REQUIRE(atTheRim < 45.0);
    }
}

TEST_CASE("a steering box with no motor in it is unassisted, exactly", "[input][ffb][rack][assist]")
{
    // The inertness proof, and it is why `PowerAssist::maximumBoost` defaults to zero rather than to
    // this car's number. An assist is something a *car* has; a bare rack is a bare rack, so every
    // case in this file that never mentions one is arithmetically untouched by the whole feature.
    const auto rack = SteeringRack{};

    auto corner = squareCorner();
    corner.tyreForce = glm::dvec3(3000.0, 0.0, 0.0);

    const auto corners = std::array<SteeredCorner, 1>{corner};
    const auto answer = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 0.0, 25.0);

    REQUIRE(answer.finite);
    REQUIRE(answer.assistForce == 0.0);
    REQUIRE(answer.driverRackForce == answer.rackForce);
    REQUIRE(answer.assistedTorque == answer.steeringTorque);
}

TEST_CASE("the power assist lightens the rack without ever reversing it", "[input][ffb][rack][assist]")
{
    auto rack = SteeringRack{};
    rack.assist.peakBoost = 2.757;

    // Both directions, because a motor that helped one way and fought the other is the single worst
    // thing a steering system can do and is not something a magnitude check would notice.
    for (const auto side : {1.0, -1.0})
    {
        auto corner = squareCorner();
        corner.tyreForce = glm::dvec3(side * 3000.0, 0.0, 0.0);

        const auto corners = std::array<SteeredCorner, 1>{corner};
        const auto answer = steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 0.0, 0.0);

        REQUIRE(answer.finite);
        CAPTURE(side, answer.rackForce, answer.assistForce, answer.driverRackForce);

        // Same sign as the load it is helping with, and never more of it than there was.
        REQUIRE(answer.assistForce * answer.rackForce > 0.0);
        REQUIRE(std::abs(answer.assistForce) < std::abs(answer.rackForce));

        // What is left for the driver is the rest of it, still pointing the same way.
        REQUIRE(answer.driverRackForce * answer.rackForce > 0.0);
        REQUIRE(std::abs(answer.assistedTorque) < std::abs(answer.steeringTorque));
        REQUIRE(answer.assistedTorque * answer.steeringTorque > 0.0);
    }
}

TEST_CASE("the boost curve sheds assist as the rack loads up", "[input][ffb][rack][assist]")
{
    auto assist = raceengine::PowerAssist{};
    assist.peakBoost = 2.757;

    SECTION("a harder-loaded rack is boosted *less*, which is what leaves the driver the limit")
    {
        // **The direction is the whole point and it was wrong once.** A curve boosting hardest where
        // the effort is sounds right, and measured on the 35 m/s sweep it took the fall in rim torque
        // past the grip peak from 50% to 34% — it eats the very cue criterion 13 is about. Every real
        // electric rack sheds assist as steering effort rises for exactly this reason.
        const auto atThePeak = std::sqrt(assist.boostKneeForce * assist.boostTaperForce);

        const auto gentle = raceengine::assistBoost(assist, 200.0, 0.0);
        const auto peak = raceengine::assistBoost(assist, atThePeak, 0.0);
        const auto hard = raceengine::assistBoost(assist, 8000.0, 0.0);

        // A hump, not a slope: held back where the signal is small, strongest in the middle, given
        // back as the fronts load up.
        REQUIRE(gentle < peak);
        REQUIRE(hard < peak);

        // And the peak is literally the field, at the force the two shaping numbers put it at.
        REQUIRE(peak == Catch::Approx(assist.peakBoost).epsilon(1e-9));

        // Past the peak it only ever falls, which is the half that protects the cue.
        REQUIRE(raceengine::assistBoost(assist, 2000.0, 0.0) > raceengine::assistBoost(assist, 4000.0, 0.0));
        REQUIRE(raceengine::assistBoost(assist, 4000.0, 0.0) > hard);
    }

    SECTION("and the same load is boosted less the faster the car is going")
    {
        const auto parked = raceengine::assistBoost(assist, 2000.0, 0.0);
        const auto rolling = raceengine::assistBoost(assist, 2000.0, 20.0);
        const auto motorway = raceengine::assistBoost(assist, 2000.0, 40.0);

        REQUIRE(rolling < parked);
        REQUIRE(motorway < rolling);

        // A fifth of it given up in the limit, half of that by the stated speed.
        REQUIRE(raceengine::assistBoost(assist, 2000.0, assist.falloffSpeed) ==
                Catch::Approx(parked * (1.0 - 0.5 * assist.speedFalloff)).epsilon(1e-9));
    }

    SECTION("no motor is no boost at any load or speed")
    {
        assist.peakBoost = 0.0;

        REQUIRE(raceengine::assistBoost(assist, 8000.0, 0.0) == 0.0);
        REQUIRE(raceengine::assistBoost(assist, 8000.0, 40.0) == 0.0);
    }
}

TEST_CASE("past the motor's capacity the driver carries every further newton", "[input][ffb][rack][assist]")
{
    // **The criterion-13 property, and the reason the cap is a physical fact rather than a clamp.**
    //
    // A flat multiplier scales the cue at the limit down along with everything else: a kerb strike
    // twice as hard arrives 0.22 times as hard, whatever it was. A real motor has a peak torque, and
    // past it the incremental gain is exactly one — so the extra reaches the driver's hands
    // undiminished. That is the difference between a wheel that tells you what happened and one that
    // tells you something happened.
    auto rack = SteeringRack{};
    rack.assist.peakBoost = 2.757;
    // Deliberately small, so the case is reached with forces this fixture can state plainly.
    rack.assist.maximumForce = 500.0;

    const auto radius = pinionRadius(rack);

    const auto driverAt = [&](const double lateral)
    {
        auto corner = squareCorner();
        corner.tyreForce = glm::dvec3(lateral, 0.0, 0.0);

        const auto corners = std::array<SteeredCorner, 1>{corner};

        return steeringRackTorque(rack, std::span<const SteeredCorner>(corners), 0.0, 0.0);
    };

    const auto first = driverAt(20000.0);
    const auto second = driverAt(24000.0);

    REQUIRE(first.finite);
    REQUIRE(second.finite);

    // Both are past it, so both get exactly the motor's capacity and no more.
    REQUIRE(std::abs(first.assistForce) == Catch::Approx(rack.assist.maximumForce).epsilon(1e-9));
    REQUIRE(std::abs(second.assistForce) == Catch::Approx(rack.assist.maximumForce).epsilon(1e-9));

    // And the whole of the difference between them reached the driver.
    const auto atTheRack = second.rackForce - first.rackForce;
    const auto atTheDriver = second.assistedTorque - first.assistedTorque;

    CAPTURE(atTheRack, atTheDriver, radius);
    REQUIRE(atTheDriver == Catch::Approx(atTheRack * radius).epsilon(1e-9));
}

TEST_CASE("this car's boost curve keeps the cue it was shaped to keep", "[input][ffb][rack][assist]")
{
    // **The acceptance test for the map, and it is deliberately not the endpoint torques.**
    //
    // What a driver reads the front axle's limit through is the *fall* in rim torque as the fronts go
    // past the grip peak: lateral force plateaus, pneumatic trail collapses, and the wheel goes
    // light. That is the entire mechanism criterion 13 tests. A boost curve is free to change how
    // heavy the wheel is; what it must not do is flatten that fall.
    //
    // The figures below are measured, unassisted, on this car's own rack at 35 m/s
    // (`./EngineTests "[.steering-geometry]"`) — the rack force and the rim torque at three points
    // through the front axle giving up.
    //
    // **The curve under test is the derived one rather than three constants.** It is built the way
    // the car builds it — `assistPlacedAtLimit` against this car's own limit rack force, which
    // `steeringLimitLoad` and `tyreAligningPeak` compute from its data — so this case fails if the
    // *derivation* stops producing a curve that keeps the cue, which is the thing that has to hold.
    // Writing the two forces here would pin a copy and let the real one drift.
    //
    // **Re-pointed 2026-08-22 when the weight distribution was corrected**, and the *unassisted* fall
    // it is measured against changed a great deal: 50.4% before, **17.1%** now. That is not the cue
    // being lost, it is the old sweep having measured the wrong thing. At 53% front the car departed
    // inside this sweep — lateral acceleration went 1.227 g, then 0.699, then 0.458, and the front
    // load collapsed from 8850 N to 2771 with the rack falling to 244 N. Half of that "fall past the
    // grip peak" was the car spinning. At the published 61.4% front it holds 1.118 g in stable
    // terminal understeer and the front saturates gently, so there is less to feel and all of what
    // is left is the front tyre. Correct, and worth a seat check.
    auto rack = raceengine::SteeringRack{};
    rack.travelPerInput = 0.070;
    rack.lockToLockDegrees = 756.0;

    const auto assist = raceengine::assistPlacedAtLimit(rack, 1916.8, 6.0);

    struct Point
    {
        double rackForce;
        double unassistedRim;
    };

    constexpr auto speed = 35.0;
    constexpr auto atPeak = Point{2192.0, 23.245};
    constexpr auto past = Point{2046.9, 21.706};
    constexpr auto wellPast = Point{1818.1, 19.280};

    const auto felt = [&](const Point& point)
    {
        return point.unassistedRim / (1.0 + raceengine::assistBoost(assist, point.rackForce, speed));
    };

    const auto unassistedFall = (atPeak.unassistedRim - wellPast.unassistedRim) / atPeak.unassistedRim;
    const auto assistedFall = (felt(atPeak) - felt(wellPast)) / felt(atPeak);

    CAPTURE(unassistedFall, assistedFall, felt(atPeak), felt(past), felt(wellPast));

    // At or above what the bare rack gives, which is the criterion. **26.4% against an unassisted
    // 17.1%** — the assist enhances the cue by a factor of 1.55, where the superseded curve managed
    // 1.04 (52.4% against 50.4%) and the version whose boost *rose* with load managed 0.67. That
    // ratio is what this case is about; the absolute figures move with the car and have twice.
    REQUIRE(unassistedFall == Catch::Approx(0.171).margin(0.01));
    REQUIRE(assistedFall > unassistedFall);
    REQUIRE(assistedFall == Catch::Approx(0.264).margin(0.02));
    REQUIRE(assistedFall / unassistedFall > 1.4);

    // Monotone all the way down, because a cue that came back up in the middle would be worse than
    // no cue at all.
    REQUIRE(felt(atPeak) > felt(past));
    REQUIRE(felt(past) > felt(wellPast));

    SECTION("and the level is anchored where the model is trustworthy")
    {
        // Six newton metres at the rim at the cornering limit. **This is the one target the level is
        // anchored on, and since 2026-08-22 it is *solved* rather than fitted**: `peakBoost` is not a
        // constant any more, it is whatever the placed shape needs to land the limit here. So the
        // check is that the solve did what it says, at its own anchor point — the limit force the
        // curve was placed against, at `assistAnchorSpeed`.
        //
        // It is stated at speed on purpose: the rack force at the limit is dominated by tyre lateral
        // force through a trail the geometry states, which is the part of this model that has been
        // checked in absolute units.
        const auto pinion = raceengine::pinionRadius(rack);
        const auto atTheLimit =
            1916.8 * pinion / (1.0 + raceengine::assistBoost(assist, 1916.8, raceengine::assistAnchorSpeed));
        REQUIRE(atTheLimit == Catch::Approx(6.0).margin(0.01));

        // And the level that came out of that solve, so a change to the shape that quietly moved it
        // is visible here rather than only in the seat. It went 2.977 -> 3.417 when the weight
        // distribution was corrected, **without anybody editing it**, which is the whole point of
        // solving it: a heavier front axle has a harder limit and the motor is sized to it.
        REQUIRE(assist.peakBoost == Catch::Approx(3.417).margin(0.02));

        // Parking at full lock, which the curve was not fitted to and lands at the bottom of the
        // 2.0-2.5 a Mk7 asks of a driver's arms. **Unmoved by the repositioning — 1.94 before and
        // 1.94 after** — which is not luck: full lock parked is 660 N, and the placed knee sits at
        // 628 N, so parking is within a few percent of the boost curve's own knee both before and
        // after. The brief that asked for the repositioning asked for parking effort to stay where
        // it was, and this is the line that says whether it did.
        //
        // The cause of it being low at all is upstream and unchanged: the unassisted parked rack
        // measures 7.0 N.m at full lock where a real car's is sixty to a hundred, because the tyre
        // model has no turn-slip torque and a parked tyre here twists instead of scrubbing, so the
        // assist is being asked to remove an effort that was never there.
        const auto parked = 7.979 / (1.0 + raceengine::assistBoost(assist, 752.4, 0.0));
        REQUIRE(parked == Catch::Approx(1.97).margin(0.1));

        // And the on-centre end, which the knee exists for. 38 degrees of lock parked reads 1.67 N.m
        // with no motor; a curve with no knee at all gave 0.41, the superseded shape gave 0.72, and
        // this one gives **1.24** — three quarters of the road, where the old one gave under half.
        //
        // That is the half of the repositioning that is easy to miss. Separating the knee from the
        // taper was done to get a limit cue, and the on-centre end came with it: measured over
        // Dominic's 201-second session the assist ratio below 1 N.m of rack torque went from 0.80 to
        // 0.99, and from 0.63 to 0.94 in the next band up. The old shape could not have both, because
        // `F/((F+k)(F+t))` is symmetric in its two forces and they were one parameter wearing two
        // names.
        const auto onCentre = 1.775 / (1.0 + raceengine::assistBoost(assist, 167.4, 0.0));
        REQUIRE(onCentre == Catch::Approx(1.30).margin(0.05));
    }
}

TEST_CASE("the boost curve's corners are second order, which is what makes them placeable",
          "[input][ffb][rack][assist]")
{
    // **The property the 2026-08-22 shape change exists to create, and it is not the one it was
    // first written down as.**
    //
    // The old load term was `F·N/((F + knee)(F + taper))` and the new one is
    // `F²/(F² + knee²) · taper²/(taper² + F²)`, normalised. The first attempt at this case asserted
    // that the old shape was symmetric under swapping the two forces and the new one was not — and
    // the case failed, because **both are symmetric**. Each is a band pass with its peak at
    // `sqrt(knee · taper)`, and swapping the two leaves either bit-identical. That is pinned below
    // rather than quietly dropped, because the wrong reason survived into three comments before a
    // test disagreed with it.
    //
    // What actually changed is the *order* of the corners: first order becomes second order, so the
    // curve falls away from its peak far faster in both directions. That is what lets the two forces
    // be placed — a limit region only about 1.5x wide in rack force needs a steep shed to get any
    // torque back across it, and with first-order corners the only way to get one is to drag the
    // peak far below the region, which drags the on-centre suppression down with it.
    auto assist = raceengine::PowerAssist{};
    assist.peakBoost = 3.0;
    assist.speedFalloff = 0.0;

    SECTION("it is a band pass, so swapping the two forces changes nothing")
    {
        auto swapped = assist;
        std::swap(swapped.boostKneeForce, swapped.boostTaperForce);

        for (const auto force : {60.0, 200.0, 900.0, 4000.0})
        {
            CAPTURE(force);
            REQUIRE(raceengine::assistBoost(assist, force, 0.0) ==
                    Catch::Approx(raceengine::assistBoost(swapped, force, 0.0)).epsilon(1e-9));
        }
    }

    SECTION("and it falls away from its peak as the square, not as the first power")
    {
        // Read as a fraction of the peak, which divides the normalisation out and is the only
        // comparison that means anything across two shapes.
        const auto peakForce = std::sqrt(assist.boostKneeForce * assist.boostTaperForce);
        const auto peak = raceengine::assistBoost(assist, peakForce, 0.0);

        const auto atTwice = raceengine::assistBoost(assist, 2.0 * peakForce, 0.0) / peak;
        const auto atFourTimes = raceengine::assistBoost(assist, 4.0 * peakForce, 0.0) / peak;

        // The superseded first-order shape gave 0.89 and 0.64 at these two points, both measured.
        CAPTURE(atTwice, atFourTimes);
        REQUIRE(atTwice < 0.72);
        REQUIRE(atFourTimes < 0.30);

        // Symmetric in log force, so the on-centre end sheds exactly as hard — which is the half
        // that buys the road feel back.
        REQUIRE(raceengine::assistBoost(assist, peakForce / 2.0, 0.0) / peak == Catch::Approx(atTwice).epsilon(1e-9));
        REQUIRE(raceengine::assistBoost(assist, peakForce / 4.0, 0.0) / peak ==
                Catch::Approx(atFourTimes).epsilon(1e-9));
    }

    // Each corner is still a *position* even though the curve is symmetric: the rise is half at the
    // knee whatever the taper is, and the shed is half at the taper whatever the knee is. The height
    // at a given force does move a little when either is touched, because the peak normalisation
    // `(taper/(taper + knee))²` has both in it — so what is checked is a ratio taken across each
    // corner, which divides that shared factor out.

    SECTION("moving the taper leaves the shape of the on-centre end alone")
    {
        // Both of these sit well below either corner, so only the rise term is doing anything.
        const auto shape = raceengine::assistBoost(assist, 60.0, 0.0) / raceengine::assistBoost(assist, 120.0, 0.0);

        auto wider = assist;
        wider.boostTaperForce = assist.boostTaperForce * 4.0;

        REQUIRE(raceengine::assistBoost(wider, 60.0, 0.0) / raceengine::assistBoost(wider, 120.0, 0.0) ==
                Catch::Approx(shape).epsilon(0.02));

        // And the far end genuinely did move, so the case is not passing by both being inert.
        REQUIRE(raceengine::assistBoost(wider, 4000.0, 0.0) > raceengine::assistBoost(assist, 4000.0, 0.0) * 1.5);
    }

    SECTION("moving the knee leaves the shape of the limit end alone")
    {
        // Both well above either corner, so only the shed term is doing anything.
        const auto shape = raceengine::assistBoost(assist, 4000.0, 0.0) / raceengine::assistBoost(assist, 8000.0, 0.0);

        auto lower = assist;
        lower.boostKneeForce = assist.boostKneeForce * 0.25;

        REQUIRE(raceengine::assistBoost(lower, 4000.0, 0.0) / raceengine::assistBoost(lower, 8000.0, 0.0) ==
                Catch::Approx(shape).epsilon(0.02));

        REQUIRE(raceengine::assistBoost(lower, 60.0, 0.0) > raceengine::assistBoost(assist, 60.0, 0.0) * 1.5);
    }

    SECTION("each corner is half of its own term, which is what makes it a position")
    {
        // The rise is half at the knee and the shed is half at the taper, by construction — so the
        // two fields are *where* each transition happens rather than two numbers that between them
        // imply one. Read through the peak, which divides the normalisation out.
        auto wide = assist;
        wide.boostKneeForce = 100.0;
        wide.boostTaperForce = 10000.0;

        const auto peak = raceengine::assistBoost(wide, std::sqrt(100.0 * 10000.0), 0.0);

        REQUIRE(raceengine::assistBoost(wide, wide.boostKneeForce, 0.0) == Catch::Approx(0.5 * peak).epsilon(0.02));
        REQUIRE(raceengine::assistBoost(wide, wide.boostTaperForce, 0.0) == Catch::Approx(0.5 * peak).epsilon(0.02));
    }

    SECTION("and the peak is still literally the field, wherever the two are put")
    {
        for (const auto knee : {200.0, 600.0, 1500.0})
        {
            for (const auto taper : {800.0, 1500.0, 6000.0})
            {
                auto placed = assist;
                placed.boostKneeForce = knee;
                placed.boostTaperForce = taper;

                CAPTURE(knee, taper);
                REQUIRE(raceengine::assistBoost(placed, std::sqrt(knee * taper), 0.0) ==
                        Catch::Approx(placed.peakBoost).epsilon(1e-9));
            }
        }
    }
}

TEST_CASE("an assist placed against a limit lands the level on its target", "[input][ffb][rack][assist]")
{
    // `assistPlacedAtLimit` is the whole of what replaced three hand-written constants, so what it
    // promises has to be checked: the shape goes where the fractions say relative to the car's limit,
    // and the level is *solved* so that the limit lands on the stated torque.
    auto rack = raceengine::SteeringRack{};
    rack.travelPerInput = 0.070;
    rack.lockToLockDegrees = 756.0;

    const auto pinion = raceengine::pinionRadius(rack);

    SECTION("the shape is placed against the limit and the level solved to the target")
    {
        for (const auto limit : {900.0, 1743.0, 3000.0})
        {
            for (const auto target : {4.0, 6.0, 9.0})
            {
                const auto placed = raceengine::assistPlacedAtLimit(rack, limit, target);
                CAPTURE(limit, target, placed.peakBoost, placed.boostKneeForce, placed.boostTaperForce);

                REQUIRE(placed.boostTaperForce == Catch::Approx(limit * raceengine::assistTaperOfLimit));
                REQUIRE(placed.boostKneeForce ==
                        Catch::Approx(placed.boostTaperForce / raceengine::assistTaperOverKnee));

                // The point of the solve: at the limit, at the anchor speed, the driver is left the
                // target. Not approximately — this is an inversion, not a fit.
                const auto felt =
                    limit * pinion / (1.0 + raceengine::assistBoost(placed, limit, raceengine::assistAnchorSpeed));
                REQUIRE(felt == Catch::Approx(target).epsilon(1e-9));
            }
        }
    }

    SECTION("a bigger car gets a bigger curve, in proportion")
    {
        // The property that makes this a derivation rather than a table: double the limit and the
        // whole shape doubles with it, so a second vehicle needs no seat session to place its assist.
        const auto small = raceengine::assistPlacedAtLimit(rack, 1000.0, 6.0);
        const auto large = raceengine::assistPlacedAtLimit(rack, 2000.0, 6.0);

        REQUIRE(large.boostKneeForce == Catch::Approx(2.0 * small.boostKneeForce));
        REQUIRE(large.boostTaperForce == Catch::Approx(2.0 * small.boostTaperForce));
    }

    SECTION("a rack already lighter than the target is left unassisted rather than fought")
    {
        // 200 N through this pinion is 2.1 N.m, already under a 6 N.m target. A motor asked to make
        // the wheel *heavier* is the one thing a power assist must never do, so the honest answer is
        // no motor at all.
        const auto placed = raceengine::assistPlacedAtLimit(rack, 200.0, 6.0);

        REQUIRE(placed.peakBoost == 0.0);
        REQUIRE(raceengine::assistBoost(placed, 200.0, 0.0) == 0.0);
    }
}
