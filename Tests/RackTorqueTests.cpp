#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

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

TEST_CASE("rack torque is plausible in absolute units for the car it came from", "[input][ffb][rack][published]")
{
    // Criterion 10, and it is the one criterion that cannot be passed by anything feeling right: the
    // trace has to be defensible in newton metres against a real vehicle. `Engine RPM` reading rad/s
    // for the whole of milestone 1 is the argument for doing this at all — a uniform scale error
    // preserves every shape a test can check and is invisible until an absolute number is compared.
    const auto car = golfGtiMk7();
    REQUIRE(car.has_value());

    const auto& hardpoints = car->corners[0].hardpoints;
    const auto solved = solveCorner(hardpoints, 0.0, 0.0);
    REQUIRE(solved.has_value());

    // A front axle at the limit: this car's front pair carries about 8 kN and a road tyre makes
    // rather more than its own load in side force at peak, so 4 kN a corner is the honest figure for
    // one wheel of a Golf on the edge of grip.
    constexpr auto sideForce = 4000.0;
    constexpr auto verticalLoad = 4000.0;
    // Pneumatic trail at peak slip, folded into Mz where it belongs. About 25 mm on a road tyre, and
    // it falls to nothing past the peak, which is the lightening a driver steers by.
    constexpr auto pneumaticTrail = 0.025;

    const auto corner = SteeredCorner{.lowerBallJoint = solved->lowerBallJoint,
                                      .upperBallJoint = solved->upperBallJoint,
                                      .steeringArm = solved->steeringArm,
                                      .rackOuter = hardpoints.steeringRackOuter,
                                      .contactPatch = solved->contactPatch,
                                      .patchNormal = glm::dvec3(0.0, 1.0, 0.0),
                                      .tyreForce = glm::dvec3(sideForce, verticalLoad, 0.0),
                                      .aligningMoment = -sideForce * pneumaticTrail};

    const auto corners = std::array<SteeredCorner, 1>{corner};
    const auto answer = steeringRackTorque(SteeringRack{}, std::span<const SteeredCorner>(corners), 0.0);

    REQUIRE(answer.finite);

    const auto atTheRim = std::abs(answer.steeringTorque) * 2.0; // both front wheels

    CAPTURE(answer.kingpinTorque[0], answer.tyreForce, answer.steeringTorque, atTheRim);

    // An unassisted rack on a front-drive hatchback at the limit sits in the low tens of newton
    // metres at the rim: enough that a driver's arms know about it, far short of the hundreds a
    // kingpin moment reads before the pinion divides it down. A figure under a couple of newton
    // metres would be a car with no steering feel at all and a figure over about fifty would be a
    // truck — both are the kind of wrong that a gain knob hides for ever.
    //
    // Note this is the *unassisted* number, which is what a wheel base should be given. The real car
    // has electric assistance between this and the driver's hands and feels a fraction of it.
    // Measured: a 168 N.m kingpin moment a corner, 1193 N at the rack, 9.94 N.m a wheel and
    // **19.9 N.m at the rim** for the pair.
    REQUIRE(atTheRim > 2.0);
    REQUIRE(atTheRim < 50.0);
}
