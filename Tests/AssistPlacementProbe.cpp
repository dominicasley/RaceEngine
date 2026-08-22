// Where this car's steering limit falls, and the power assist placed against it.
// `./EngineTests "[.assist-placement]"`.
//
// **The whole point of the exercise, printed.** Until 2026-08-22 the assist's knee and taper were
// two constants in `SimulatedCar` — 500 N and 2500 N — and the only way to find out whether they
// were in the right place was to drive the car and read a trace. That is a seat session per vehicle,
// and it is what this replaces: everything below is computed from the car's own data at load time,
// and the numbers it prints are the ones the game will actually run with.
//
// The chain, in order, with the module each link lives in:
//
//   `steeringLimitLoad`   physics   what the outside front carries at the limit — a fixed point
//                                   between load transfer and the friction that load transfer costs
//   `tyreAligningPeak`    physics   the slip angle where aligning moment turns over, and the force
//                                   and moment there — the steering limit, as a tyre property
//   `steeringRackTorque`  input     those forces through the kingpin and the tie rod, into newtons
//                                   at the rack — the *same* function the tick calls every frame
//   `assistPlacedAtLimit` input     the knee and the taper as fractions of that
//
// Nothing here re-derives anything: a second copy of the kingpin arithmetic would be free to drift
// from the one the car runs on, and the assist would then be placed against a limit the car does not
// have.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <tuple>

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;
import raceengine;

namespace
{

constexpr auto degrees = 57.29577951308232;

}

TEST_CASE("where the steering limit falls, and the assist placed against it", "[.assist-placement]")
{
    const auto built = raceengine::golfGtiMk7();
    REQUIRE(built.has_value());
    const auto setup = built.value();

    const auto loads = raceengine::steeringLimitLoad(setup);

    std::printf("\n=== the front axle at the limit ===\n");
    std::printf("  static per wheel   %8.1f N\n", loads.staticPerWheel);
    std::printf("  outside at limit   %8.1f N   (%.2fx static)\n", loads.outside,
                loads.outside / std::max(loads.staticPerWheel, 1e-9));
    std::printf("  inside at limit    %8.1f N\n", loads.inside);
    std::printf("  MEASURED on Dominic's 2026-08-22 Bathurst session: outside front median 5851 N\n"
                "  through the slip band where its aligning moment peaks — but that session was driven\n"
                "  on the car BEFORE the weight distribution was corrected, whose static front corner\n"
                "  was 3503 N against this one's %.0f. Against that car the derivation gave 6375 N,\n"
                "  9%% high. There is no measurement of the corrected car yet.\n",
                loads.staticPerWheel);

    const auto& tyre = setup.corners[0].tyre;

    std::printf("\n=== where the tyre's aligning moment turns over ===\n");
    std::printf("  %10s %12s %12s %12s\n", "Fz [N]", "slip [deg]", "|Mz| [Nm]", "Fy [N]");
    for (const auto load : {loads.staticPerWheel, loads.outside, loads.inside})
    {
        const auto peak = raceengine::tyreAligningPeak(tyre, load);
        std::printf("  %10.0f %12.2f %12.2f %12.1f\n", load, peak.slipAngle * degrees, std::abs(peak.aligningMoment),
                    peak.lateralForce);
    }

    auto corners = std::array<raceengine::SteeredCorner, raceengine::steeredCornerLimit>{};
    for (auto index = std::size_t{0}; index < raceengine::steeredCornerLimit; index++)
    {
        const auto& hardpoints = setup.corners[index].hardpoints;
        const auto solved = raceengine::solveCorner(hardpoints, 0.0, 0.0);
        REQUIRE(solved.has_value());

        const auto load = index == 0 ? loads.outside : loads.inside;
        const auto limit = raceengine::tyreAligningPeak(tyre, load);

        corners[index] = raceengine::SteeredCorner{.lowerBallJoint = solved->lowerBallJoint,
                                                   .upperBallJoint = solved->upperBallJoint,
                                                   .steeringArm = solved->steeringArm,
                                                   .rackOuter = hardpoints.steeringRackOuter,
                                                   .contactPatch = solved->contactPatch,
                                                   .patchNormal = glm::dvec3(0.0, 1.0, 0.0),
                                                   .tyreForce = glm::dvec3(limit.lateralForce, load, 0.0),
                                                   .aligningMoment = limit.aligningMoment};
    }

    auto bare = raceengine::SteeringRack{};
    bare.travelPerInput = setup.rackTravelPerInput;
    bare.lockToLockDegrees = 756.0;
    bare.friction = 0.0;
    bare.damping = 0.0;
    bare.assist = raceengine::PowerAssist{};

    const auto atLimit =
        raceengine::steeringRackTorque(bare, std::span<const raceengine::SteeredCorner>(corners), 0.0, 0.0);
    REQUIRE(atLimit.finite);

    const auto pinion = raceengine::pinionRadius(bare);
    const auto limitForce = std::abs(atLimit.rackForce);

    std::printf("\n=== through the linkage, into the rack ===\n");
    std::printf("  kingpin torque     %8.2f / %.2f Nm (outside / inside)\n", atLimit.kingpinTorque[0],
                atLimit.kingpinTorque[1]);
    std::printf("  pinion radius      %8.5f m\n", pinion);
    std::printf("  LIMIT RACK FORCE   %8.1f N = %.2f Nm at the rim\n", limitForce, limitForce * pinion);
    std::printf("  MEASURED on the pre-correction car: the outside front's aligning moment peaks at\n"
                "  17.15 Nm of rack torque, which is 1617 N, and the derivation gave 1743 N there —\n"
                "  7.8%% high, the geometry between load and rack force being exact. This car's front\n"
                "  axle carries more, so its limit is higher and unmeasured; the +%.1f%% against 1617\n"
                "  below is a corrected car against an uncorrected lap and is NOT a validation.\n",
                100.0 * (limitForce / 1617.0 - 1.0));

    const auto placed = raceengine::assistPlacedAtLimit(bare, limitForce, 6.0);
    const auto boostPeak = std::sqrt(placed.boostKneeForce * placed.boostTaperForce);

    std::printf("\n=== the assist placed against it ===\n");
    std::printf("  knee               %8.1f N = %5.2f Nm   (taper / %.1f)\n", placed.boostKneeForce,
                placed.boostKneeForce * pinion, raceengine::assistTaperOverKnee);
    std::printf("  taper              %8.1f N = %5.2f Nm   (%.2f x the limit)\n", placed.boostTaperForce,
                placed.boostTaperForce * pinion, raceengine::assistTaperOfLimit);
    std::printf("  boost peaks at     %8.1f N = %5.2f Nm\n", boostPeak, boostPeak * pinion);
    std::printf("  peak boost         %8.3f       solved from the 6.00 Nm target at the limit\n", placed.peakBoost);
    std::printf("  SUPERSEDED: knee 500 N (5.30 Nm), taper 2500 N (26.51 Nm), boost peaking at\n"
                "  1118 N (11.86 Nm) — which is the *bottom* of the limit region, so the motor was at\n"
                "  its most compressive across the whole of the cue and the taper never did any work.\n");

    std::printf("\n=== the curve, in newton metres at the rim ===\n");
    std::printf("  %10s %10s %10s %10s\n", "T_rack", "boost", "assisted", "ratio");
    for (const auto torque : {0.3, 0.7, 1.5, 3.0, 6.0, 10.0, 14.0, 17.0, 20.0, 30.0, 45.0})
    {
        const auto force = torque / pinion;
        const auto boost = raceengine::assistBoost(placed, force, 25.0);
        const auto assisted = torque / (1.0 + boost);

        std::printf("  %10.1f %10.3f %10.2f %10.2f\n", torque, boost, assisted, assisted / torque);
    }

    // The two the level is anchored on, both parked, so the speed schedule is out of the way.
    std::printf("\n=== where the level lands ===\n");
    for (const auto [what, torque, force] :
         {std::tuple{"parking, full lock", 7.00, 659.8}, std::tuple{"parked, 38 deg lock", 1.666, 157.1}})
    {
        std::printf("  %-22s %6.2f Nm unassisted -> %.2f Nm\n", what, torque,
                    torque / (1.0 + raceengine::assistBoost(placed, force, 0.0)));
    }
}
