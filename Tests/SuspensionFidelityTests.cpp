// The mechanisms `docs/suspension-fidelity-brief.md` names, and what each of them is worth.
//
// Every case here is about a *mechanism* rather than a number, which is the brief's own framing.
// What they gate is that the built ones do what they claim, that the ones meant to be inert are
// inert to the bit, and that the two structural facts the measurements turned up cannot quietly stop
// being true. The printed A/B lives in `[.driveline-path]` and `[.driveline-reaction]`; this is the
// part `ctest` runs.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine.physics;

using raceengine::applyComplianceCamber;
using raceengine::applyComplianceRecession;
using raceengine::cornerCount;
using raceengine::CornerHardpoints;
using raceengine::CornerSetup;
using raceengine::CornerSide;
using raceengine::damperDampingCoefficient;
using raceengine::dropLinkElementOf;
using raceengine::dropLinkStated;
using raceengine::golfGtiMk7;
using raceengine::golfMk7FrontCorner;
using raceengine::golfMk7RearCorner;
using raceengine::outboardSign;
using raceengine::placeholderCorner;
using raceengine::solveAntiRollBar;
using raceengine::solveCornerWithJacobian;
using raceengine::solveDamperForce;
using raceengine::SuspensionState;
using raceengine::validateCornerSetup;

namespace
{

[[nodiscard]] SuspensionState solvedAt(const CornerHardpoints& hardpoints, const double angle)
{
    auto state = solveCornerWithJacobian(hardpoints, angle, 0.0);
    REQUIRE(state.has_value());

    return state.value();
}

// A placeholder corner is the only geometry in this project that states a drop link, so it is what
// the geometric bar is exercised on. That is a data fact rather than a choice — see the case below.
[[nodiscard]] CornerSetup barCorner(const CornerSide side, const double rate)
{
    auto corner = CornerSetup{};
    corner.hardpoints = placeholderCorner(side, 1.35);
    corner.antiRollRate = rate;

    return corner;
}

[[nodiscard]] CornerSetup withoutDropLink(CornerSetup corner)
{
    corner.hardpoints.antiRollBarChassis = glm::dvec3(0.0);
    corner.hardpoints.antiRollBarWishbone = glm::dvec3(0.0);

    return corner;
}

} // namespace

// --- item 2: the anti-roll bar's geometry ------------------------------------------------------

TEST_CASE("which corners state a drop link, and which do not", "[physics][suspension][antiroll]")
{
    // **The brief said these hardpoints were authored on every corner. They are not.**
    // `suspensions.ini` states `ARB FRONT 34000 / REAR 15000` and no coordinates at all, so the Golf
    // has none on either axle — which is why the bar geometry is opt-in and why switching it on for
    // the real car is blocked on data, exactly as bushing compliance is. Pinned so that the day
    // somebody sources the drop-link points, this case fails and says so.
    REQUIRE_FALSE(dropLinkStated(golfMk7FrontCorner(CornerSide::Left)));
    REQUIRE_FALSE(dropLinkStated(golfMk7FrontCorner(CornerSide::Right)));
    REQUIRE_FALSE(dropLinkStated(golfMk7RearCorner(CornerSide::Left)));
    REQUIRE_FALSE(dropLinkStated(golfMk7RearCorner(CornerSide::Right)));

    // The placeholder corner does state one, and always has — it was authored and then read by
    // nothing for as long as the linkage has existed.
    REQUIRE(dropLinkStated(placeholderCorner(CornerSide::Left, 1.35)));

    const auto link = dropLinkElementOf(placeholderCorner(CornerSide::Left, 1.35));
    REQUIRE(link.chassis != link.wishbone);
}

TEST_CASE("a corner with no drop link keeps the bar it always had", "[physics][suspension][antiroll]")
{
    // The wheel-referred model, factor for factor. This is the expression that was inline in the
    // force pass, and every car in this project is still on it — so it has to come back out of the
    // new function bit for bit rather than merely close.
    const auto self = withoutDropLink(barCorner(CornerSide::Left, 34000.0));
    const auto across = withoutDropLink(barCorner(CornerSide::Right, 34000.0));

    const auto here = solvedAt(self.hardpoints, 0.03);
    const auto there = solvedAt(across.hardpoints, -0.02);

    const auto bar = solveAntiRollBar(self, here, across, there);

    REQUIRE_FALSE(bar.geometric);
    REQUIRE(bar.linkForce == 0.0);
    REQUIRE(bar.wheelForce == 34000.0 * (there.wheelTravel - here.wheelTravel));
}

TEST_CASE("a bar with no rate makes no force whichever model it is on", "[physics][suspension][antiroll]")
{
    const auto self = barCorner(CornerSide::Left, 0.0);
    const auto across = barCorner(CornerSide::Right, 0.0);

    const auto bar =
        solveAntiRollBar(self, solvedAt(self.hardpoints, 0.03), across, solvedAt(across.hardpoints, -0.02));

    REQUIRE(bar.wheelForce == 0.0);
    REQUIRE(bar.linkForce == 0.0);
    REQUIRE_FALSE(bar.geometric);
}

TEST_CASE("the geometric bar carries the authored wheel rate", "[physics][suspension][antiroll]")
{
    // **The trap the brief names, and the whole reason the referral is at the design position.**
    // `antiRollRate` is documented as a rate at the wheel, and every car states it that way; the
    // moment the bar gains a motion ratio that number changes meaning unless something converts it.
    // `k_link = k_wheel / ratio²` is that conversion, and it is evaluated at design — so a small
    // differential about the design position has to reproduce the wheel-referred answer.
    const auto self = barCorner(CornerSide::Left, 34000.0);
    const auto across = barCorner(CornerSide::Right, 34000.0);

    for (const auto differential : std::vector<double>{0.002, 0.005, 0.01})
    {
        const auto here = solvedAt(self.hardpoints, differential);
        const auto there = solvedAt(across.hardpoints, -differential);

        const auto geometric = solveAntiRollBar(self, here, across, there);
        const auto referred = solveAntiRollBar(withoutDropLink(self), here, withoutDropLink(across), there);

        REQUIRE(geometric.geometric);
        REQUIRE_FALSE(referred.geometric);

        CAPTURE(differential, geometric.wheelForce, referred.wheelForce);

        // Same sign and same size. The residual is the ratio's own variation across the travel the
        // corner has moved, which is the fidelity the geometry buys and is therefore not asked to be
        // zero — it is asked to be small at small displacements, which is what "the authored number
        // still means what it says" reduces to.
        REQUIRE(geometric.wheelForce == Catch::Approx(referred.wheelForce).epsilon(0.05));
    }
}

TEST_CASE("the bar's link force is exactly equal and opposite across its axle", "[physics][suspension][antiroll]")
{
    // A torsion bar is one internal force, so the two ends carry the same load. **Exact**, and it is
    // exact by construction rather than by luck: the referral divides by *both* corners' design
    // ratios, so the two ends compute the same link rate whatever their geometries are, and the
    // difference in link displacement is the exact negation of itself.
    const auto self = barCorner(CornerSide::Left, 34000.0);
    const auto across = barCorner(CornerSide::Right, 34000.0);

    const auto here = solvedAt(self.hardpoints, 0.04);
    const auto there = solvedAt(across.hardpoints, -0.015);

    const auto left = solveAntiRollBar(self, here, across, there);
    const auto right = solveAntiRollBar(across, there, self, here);

    REQUIRE(left.linkForce != 0.0);
    REQUIRE(left.linkForce == -right.linkForce);

    // And at the *wheel* they are not, which is the correction rather than a defect: the two corners
    // sit at different points in their travel and convert the same link force differently. A bar with
    // no motion ratio cannot express that.
    REQUIRE(left.wheelForce != -right.wheelForce);
}

TEST_CASE("the bar makes no force in pure heave", "[physics][suspension][antiroll]")
{
    // Both wheels moving together is what a bar must ignore, and it is the statement a sign error in
    // the difference of link displacements cannot survive.
    const auto self = barCorner(CornerSide::Left, 34000.0);
    const auto across = barCorner(CornerSide::Right, 34000.0);

    const auto here = solvedAt(self.hardpoints, 0.05);
    const auto there = solvedAt(across.hardpoints, 0.05);

    const auto bar = solveAntiRollBar(self, here, across, there);

    REQUIRE(bar.geometric);
    REQUIRE(bar.linkForce == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(bar.wheelForce == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE("the bar pushes back on whichever corner is deeper into bump", "[physics][suspension][antiroll]")
{
    // The sign, on the geometric model, stated where it is legible: the wheel that is further into
    // bump gets a *negative* equivalent wheel force, which pushes it back out. A bar wired backwards
    // would be adding to the roll it exists to resist.
    const auto self = barCorner(CornerSide::Left, 34000.0);
    const auto across = barCorner(CornerSide::Right, 34000.0);

    const auto here = solvedAt(self.hardpoints, 0.04);
    const auto there = solvedAt(across.hardpoints, -0.04);

    REQUIRE(here.wheelTravel > there.wheelTravel);

    const auto bar = solveAntiRollBar(self, here, across, there);

    REQUIRE(bar.geometric);
    REQUIRE(bar.wheelForce < 0.0);
}

TEST_CASE("half a drop link is refused at load time", "[physics][suspension][antiroll]")
{
    // A corner that states one end and leaves the other on the origin passes `dropLinkStated` and
    // then describes a link that sweeps most of a metre for a few millimetres of wheel. The force
    // path falls back rather than dividing by it, so without this the mistake is silent.
    auto corner = CornerSetup{};
    corner.hardpoints = placeholderCorner(CornerSide::Left, 1.35);
    corner.springRate = 73000.0;
    corner.springFreeLength = 0.5;
    corner.bumpStop = raceengine::TravelStop{.gap = 0.020, .rate = 900000.0, .progression = 3.0, .damping = 40000.0};
    corner.droopStop = raceengine::TravelStop{.gap = 0.020, .rate = 600000.0, .progression = 3.0, .damping = 30000.0};

    REQUIRE(validateCornerSetup(corner).has_value());

    corner.hardpoints.antiRollBarChassis = glm::dvec3(0.0);

    const auto refused = validateCornerSetup(corner);
    REQUIRE_FALSE(refused.has_value());
}

// --- items 3 and 6: the patch is not a material point, and what follows from that ---------------

TEST_CASE("the constructed contact patch is not a material point of the upright", "[physics][suspension][loadpath]")
{
    // **The structural fact the whole of item 3 turns on**, pinned here rather than left in a probe.
    //
    // `SuspensionState::contactPatch` is the wheel centre plus the tyre's radius along the *world's*
    // down direction projected into the wheel's plane, so it depends on the spin axis and on nothing
    // else about how the upright is standing. Turn the upright about that axis and the constructed
    // patch does not move; the real one rolls by a radius times the angle.
    //
    // The consequence is exactly this: the difference between the two Jacobians is **purely
    // lateral**. That is why the load-path brief found the patch's side-view ratio and the wheel
    // centre's agreeing to every digit it printed, and read it as evidence that the spin term was
    // negligible. It is not evidence about that term at all.
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto state = solvedAt(built->corners[index].hardpoints, 0.0);
        const auto difference = state.patchPerAngle - state.wheelCentrePerAngle;

        CAPTURE(index, difference.x, difference.y, difference.z);

        REQUIRE(std::abs(difference.y) < 1e-12);
        REQUIRE(std::abs(difference.z) < 1e-12);

        // And the lateral part is real rather than also zero, so this is a statement about the
        // construction and not about a corner that happens not to move.
        REQUIRE(std::abs(difference.x) > 1e-3);
    }
}

TEST_CASE("the wheel centre's vertical rate is the travel Jacobian", "[physics][suspension][loadpath]")
{
    // Two fields differenced from the same pair of solves, so they cannot disagree. Asserted because
    // the generalised inertia now reads one of them and the force path the other, and a divergence
    // between them would be silent.
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto state = solvedAt(built->corners[index].hardpoints, 0.0);

        REQUIRE(state.wheelCentrePerAngle.y == state.travelPerAngle);
    }
}

TEST_CASE("the upright's angular rate is the rotation the solve actually performs", "[physics][suspension][loadpath]")
{
    // `uprightRatePerAngle` is extracted from the vector part of a quaternion difference, which is a
    // small-angle reading. This is the check that it is the real rotation: carrying the design
    // orientation forward by it has to land on the orientation the solver produces a step later, to
    // the order the linearisation allows.
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    constexpr auto step = 1e-4;

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& hardpoints = built->corners[index].hardpoints;

        const auto here = solvedAt(hardpoints, 0.0);
        const auto there = solvedAt(hardpoints, step);

        const auto rate = here.uprightRatePerAngle;
        const auto magnitude = glm::length(rate);
        REQUIRE(magnitude > 1e-6);

        const auto carried = glm::angleAxis(magnitude * step, rate / magnitude) * here.uprightOrientation;
        const auto residual = glm::angle(glm::normalize(carried * glm::conjugate(there.uprightOrientation)));

        CAPTURE(index, magnitude, residual);

        // Second order in the step, so a thousandth of the rotation performed is generous and a
        // wrong axis or a dropped sign is nowhere near it.
        REQUIRE(residual < 1e-3 * magnitude * step + 1e-12);
    }
}

TEST_CASE("the drive term collapses the patch line onto the wheel-centre line",
          "[physics][suspension][loadpath][driveline]")
{
    // The identity that says the shaft term's sign is right, and the only check in the suite that
    // does. With the shaft torque coming from the chassis, the corner's generalised force from a
    // steady tractive force is the material patch Jacobian plus `T · dTheta_spin/dq` with
    // `T = Fx · r` — and the textbook answer for an inboard differential is that the whole thing is
    // the wheel-centre line. A sign error anywhere in the chain breaks this and nothing else would
    // notice.
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& hardpoints = built->corners[index].hardpoints;
        const auto state = solvedAt(hardpoints, 0.0);

        const auto belowCentre = state.contactPatch - state.wheelCentre;
        const auto material = state.wheelCentrePerAngle + glm::cross(state.uprightRatePerAngle, belowCentre);

        // The axis a positive wheel speed turns about, which is the car's left on **every** corner —
        // not the outboard-pointing axis the wheel is posed with. Both wheels of a car going forward
        // turn the same way.
        const auto spin = glm::dot(state.uprightRatePerAngle, state.uprightOrientation * glm::dvec3(1.0, 0.0, 0.0));

        CAPTURE(index, material.z, spin, state.wheelCentrePerAngle.z);

        REQUIRE(material.z + hardpoints.wheelRadius * spin == Catch::Approx(state.wheelCentrePerAngle.z).margin(1e-12));

        // And the term is not negligible on this car, which is the answer to the question the brief
        // stopped on: the spin part is most of the longitudinal channel.
        REQUIRE(std::abs(hardpoints.wheelRadius * spin) > 0.3 * std::abs(state.wheelCentrePerAngle.z));
    }
}

TEST_CASE("carrying the patch at the loaded radius moves the lateral channel and nothing else",
          "[physics][suspension][loadpath]")
{
    // Item 6, and the reason it was bundled rather than done on its own. The road is at the loaded
    // radius, so that is where the force is applied; scaling the radius interpolates the Jacobians.
    // Measured at 15 mm of deflection the scrub ratio falls a few per cent and the side-view ratio
    // does not move at all, because the part of the patch's motion the radius scales is the radius
    // turning with camber and that is purely lateral.
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    constexpr auto deflection = 0.015;

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto& hardpoints = built->corners[index].hardpoints;
        const auto state = solvedAt(hardpoints, 0.0);

        const auto belowCentre = state.contactPatch - state.wheelCentre;
        const auto scale = (hardpoints.wheelRadius - deflection) / hardpoints.wheelRadius;

        const auto free = state.wheelCentrePerAngle + glm::cross(state.uprightRatePerAngle, belowCentre);
        const auto loaded = state.wheelCentrePerAngle + glm::cross(state.uprightRatePerAngle, belowCentre * scale);

        CAPTURE(index, free.x, loaded.x, free.z, loaded.z);

        REQUIRE(loaded.y == Catch::Approx(free.y).margin(1e-12));
        REQUIRE(std::abs(loaded.x) < std::abs(free.x));

        // The longitudinal channel moves too here, and that is the difference from the *constructed*
        // patch: once the point is carried by the upright, shortening the radius shortens the spin
        // term as well. Both channels scale, and neither is left behind.
        REQUIRE(std::abs(loaded.z - state.wheelCentrePerAngle.z) < std::abs(free.z - state.wheelCentrePerAngle.z));
    }
}

// --- item 4: the corner's generalised inertia ---------------------------------------------------

TEST_CASE("the corner's generalised inertia rises when the whole Jacobian is used", "[physics][suspension][inertia]")
{
    // `m·|dC/dq|²` against `m·(dC/dq).y²`. The brief predicted about 3% from the *patch's* in-plane
    // components; the wheel centre's are much smaller, because most of the patch's lateral motion is
    // the tyre's radius turning with camber rather than the hub going anywhere. Measured it is half a
    // per cent at the front and a tenth of one at the rear — small, correct, and free.
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        const auto state = solvedAt(built->corners[index].hardpoints, 0.0);

        const auto whole = glm::dot(state.wheelCentrePerAngle, state.wheelCentrePerAngle);
        const auto vertical = state.travelPerAngle * state.travelPerAngle;

        CAPTURE(index, whole, vertical, whole / vertical - 1.0);

        REQUIRE(whole > vertical);
        REQUIRE(whole / vertical - 1.0 < 0.02);
    }
}

TEST_CASE("a wheel that only moves vertically has the same inertia either way", "[physics][suspension][inertia]")
{
    // The inertness statement: where the two models are the same model, they are the same number.
    // Built rather than found — a corner whose wheel centre sits on the lower arm's own swing axis
    // plane and whose in-plane rates vanish is not a car, so this asserts the arithmetic directly on
    // a Jacobian with no in-plane content.
    const auto vertical = glm::dvec3(0.0, 0.3458, 0.0);

    REQUIRE(glm::dot(vertical, vertical) == vertical.y * vertical.y);
}

// --- item 5: damper friction --------------------------------------------------------------------

namespace
{

[[nodiscard]] CornerSetup damperCorner(const double friction)
{
    auto corner = CornerSetup{};
    corner.hardpoints = placeholderCorner(CornerSide::Left, 1.35);
    corner.damper = raceengine::linearDamper(4200.0, 7600.0);
    corner.damperFriction = friction;

    return corner;
}

} // namespace

TEST_CASE("a damper with no friction stated runs the curve it always ran", "[physics][suspension][damper]")
{
    // Inert to the bit, and branched rather than added for exactly that reason: `x + 0.0 · tanh(y)`
    // is not `x` for every x, and every car in this project states no friction.
    const auto plain = damperCorner(0.0);
    const auto state = solvedAt(plain.hardpoints, 0.02);

    for (const auto rate : std::vector<double>{-2.0, -0.4, -0.001, 0.0, 0.001, 0.4, 2.0})
    {
        const auto solved = solveDamperForce(plain, state, rate);

        REQUIRE(solved.force == plain.damper.at(solved.velocity));
    }
}

TEST_CASE("damper friction opposes the shaft and saturates", "[physics][suspension][damper]")
{
    const auto plain = damperCorner(0.0);
    const auto rough = damperCorner(120.0);
    const auto state = solvedAt(rough.hardpoints, 0.02);

    // At rest there is no friction force at all, which is what makes this a dead band rather than a
    // preload: a stationary damper is not pushing anything.
    REQUIRE(solveDamperForce(rough, state, 0.0).force == Catch::Approx(0.0).margin(1e-12));

    for (const auto rate : std::vector<double>{-2.0, -0.4, 0.4, 2.0})
    {
        const auto with = solveDamperForce(rough, state, rate);
        const auto without = solveDamperForce(plain, state, rate);

        CAPTURE(rate, with.velocity, with.force, without.force);

        // Along the shaft's own motion, which for a velocity well past the smoothing width is the
        // whole of the stated friction.
        REQUIRE(with.force - without.force == Catch::Approx(std::copysign(120.0, with.velocity)).epsilon(1e-6));
    }
}

TEST_CASE("damper friction reaches the implicit damping coefficient", "[physics][suspension][damper]")
{
    // It has to, or the integration fights it: the friction's slope at zero velocity is
    // `friction / smoothing`, which is the largest number in the corner's damping and the entire
    // reason the term is solved rather than stepped towards.
    const auto plain = damperCorner(0.0);
    const auto rough = damperCorner(120.0);
    const auto state = solvedAt(rough.hardpoints, 0.02);

    const auto still = solveDamperForce(rough, state, 0.0);
    const auto plainStill = solveDamperForce(plain, state, 0.0);

    const auto added = damperDampingCoefficient(rough, still) - damperDampingCoefficient(plain, plainStill);
    const auto expected = still.lengthPerAngle * still.lengthPerAngle * (120.0 / rough.damperFrictionSpeed);

    CAPTURE(added, expected);

    REQUIRE(added == Catch::Approx(expected).epsilon(1e-6));

    // And a damper with none stated is the coefficient it always was, to the bit.
    REQUIRE(damperDampingCoefficient(plain, plainStill) ==
            plainStill.lengthPerAngle * plainStill.lengthPerAngle *
                std::max(0.0, (plain.damper.at(1e-4) - plain.damper.at(-1e-4)) / 2e-4));
}

// --- item 1's other half: lateral-force compliance camber ---------------------------------------

namespace
{

// The published figure exactly as the car data states it: Kawata, Kouno & Sakuma, Trans. JSME
// 89(919) 2023, Table 2 — median 0.17 deg/kN across four front axles — converted to radians per
// newton the same way the setup sheet converts it.
constexpr auto camberPerNewton = 0.17 * 0.017453292519943295 / 1000.0;

} // namespace

TEST_CASE("a compliance camber of nothing leaves the solved corner untouched to the bit",
          "[physics][suspension][compliance]")
{
    // The vehicle step branches on the coefficient, so a car stating none never calls this at all —
    // but the stronger statement holds too: even called with a zero angle, the identity rotation and
    // the re-read reproduce the same bits, because `readOffWheel` is one statement of what the
    // orientation decides and the orientation has not moved.
    const auto hardpoints = golfMk7FrontCorner(CornerSide::Left);
    const auto untouched = solvedAt(hardpoints, 0.01);

    auto twisted = untouched;
    applyComplianceCamber(hardpoints, twisted, 0.0);

    REQUIRE(twisted.uprightOrientation.w == untouched.uprightOrientation.w);
    REQUIRE(twisted.uprightOrientation.x == untouched.uprightOrientation.x);
    REQUIRE(twisted.uprightOrientation.y == untouched.uprightOrientation.y);
    REQUIRE(twisted.uprightOrientation.z == untouched.uprightOrientation.z);
    REQUIRE(twisted.camber == untouched.camber);
    REQUIRE(twisted.toe == untouched.toe);
    REQUIRE(twisted.contactPatch == untouched.contactPatch);
    REQUIRE(twisted.halfTrack == untouched.halfTrack);
}

TEST_CASE("a stated compliance camber produces the derived lean at a known force", "[physics][suspension][compliance]")
{
    // 3.2 kN of lateral force at the coefficient the car states is 0.544 degrees of lean. The angle
    // is applied about the chassis's forward axis, so what `readOffWheel` reports moves by
    // `-outboard x angle` — the sign the spin-axis construction folds in — and the toe does not move
    // at all, because a rotation about +z leaves every vector's z component alone. That last part is
    // the difference between camber compliance and steer compliance, asserted rather than assumed.
    const auto angle = camberPerNewton * 3200.0;

    for (const auto side : {CornerSide::Left, CornerSide::Right})
    {
        const auto hardpoints = golfMk7FrontCorner(side);
        const auto untouched = solvedAt(hardpoints, 0.0);

        auto twisted = untouched;
        applyComplianceCamber(hardpoints, twisted, angle);

        CAPTURE(outboardSign(side), untouched.camber, twisted.camber);

        REQUIRE(twisted.camber - untouched.camber == Catch::Approx(-outboardSign(side) * angle).epsilon(0.01));
        REQUIRE(twisted.toe == Catch::Approx(untouched.toe).margin(1e-12));

        // The hub does not move — the wheel leans about it, which is the whole of what is sourced.
        REQUIRE(twisted.wheelCentre == untouched.wheelCentre);
    }
}

TEST_CASE("the loaded outside wheel leans out of the turn and its patch tucks under",
          "[physics][suspension][compliance]")
{
    // The sign case, stated on the physical situation rather than on the arithmetic. In a left turn
    // the road pushes every patch towards the turn centre, +x, so the vehicle step hands this corner
    // a positive angle. On the right — outside, loaded — wheel that must read as the top leaning
    // away from the car (positive SAE camber, out of a left turn: the adverse direction) with the
    // patch displacing vehicle-inward, which is the source's own sign convention for a positive
    // coefficient.
    const auto hardpoints = golfMk7FrontCorner(CornerSide::Right);
    const auto untouched = solvedAt(hardpoints, 0.0);

    const auto outsideWheelLoad = 4200.0;
    auto twisted = untouched;
    applyComplianceCamber(hardpoints, twisted, camberPerNewton * outsideWheelLoad);

    CAPTURE(untouched.camber, twisted.camber, untouched.contactPatch.x, twisted.contactPatch.x);

    REQUIRE(twisted.camber > untouched.camber);
    REQUIRE(twisted.contactPatch.x > untouched.contactPatch.x);

    // And the same force leans the inside wheel the same way in space — towards the outside of the
    // turn — which its own outboard sign reads as negative camber. Both wheels lean out of the turn;
    // neither gains the camber a designer would want.
    const auto inside = golfMk7FrontCorner(CornerSide::Left);
    const auto insideUntouched = solvedAt(inside, 0.0);

    auto insideTwisted = insideUntouched;
    applyComplianceCamber(inside, insideTwisted, camberPerNewton * 1400.0);

    REQUIRE(insideTwisted.camber < insideUntouched.camber);
}

TEST_CASE("which axle states compliance camber and what it states", "[physics][suspension][compliance]")
{
    // Pinned the way the drop-link facts are, so the day the figure moves — or the day somebody
    // sources a rear measurement — a test says so. The rear is deliberately zero: the JSME campaign
    // measured front axles only, and Heissing/Ersoy's rear < 1.0 deg/kN is a design target rather
    // than a measurement.
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    REQUIRE(built->corners[0].lateralForceCamber == camberPerNewton);
    REQUIRE(built->corners[1].lateralForceCamber == camberPerNewton);
    REQUIRE(built->corners[2].lateralForceCamber == 0.0);
    REQUIRE(built->corners[3].lateralForceCamber == 0.0);
}

// --- item 1's translation third: longitudinal recession -----------------------------------------

TEST_CASE("a recession of nothing leaves the solved corner untouched to the bit",
          "[physics][suspension][compliance]")
{
    // The vehicle step branches on the coefficient, so a car stating none never calls this at all —
    // and even called with a zero displacement, adding 0.0 to a real coordinate and re-reading the
    // unmoved centre reproduces the same bits.
    const auto hardpoints = golfMk7FrontCorner(CornerSide::Left);
    const auto untouched = solvedAt(hardpoints, 0.01);

    auto receded = untouched;
    applyComplianceRecession(hardpoints, receded, 0.0);

    REQUIRE(receded.wheelCentre == untouched.wheelCentre);
    REQUIRE(receded.contactPatch == untouched.contactPatch);
    REQUIRE(receded.camber == untouched.camber);
    REQUIRE(receded.toe == untouched.toe);
    REQUIRE(receded.halfTrack == untouched.halfTrack);
}

TEST_CASE("a stated recession produces the derived displacement at a known force",
          "[physics][suspension][compliance]")
{
    // 3 kN of braking force at the design band's middle — 6 mm/kN — is 18 mm of rearward
    // displacement, linear and exact. The braking force on the car points rearward, so the vehicle
    // step hands this seam a negative displacement; the hub moves by exactly it, the constructed
    // patch moves with the hub, and nothing the orientation decides moves at all.
    constexpr auto recessionPerNewton = 6.0 * 1.0e-3 / 1000.0;
    const auto displacement = recessionPerNewton * -3000.0;

    for (const auto side : {CornerSide::Left, CornerSide::Right})
    {
        const auto hardpoints = golfMk7FrontCorner(side);
        const auto untouched = solvedAt(hardpoints, 0.0);

        auto receded = untouched;
        applyComplianceRecession(hardpoints, receded, displacement);

        CAPTURE(outboardSign(side), untouched.wheelCentre.z, receded.wheelCentre.z);

        REQUIRE(receded.wheelCentre.z - untouched.wheelCentre.z == Catch::Approx(-0.018).epsilon(1e-9));
        REQUIRE(receded.contactPatch.z - untouched.contactPatch.z == Catch::Approx(-0.018).epsilon(1e-9));

        REQUIRE(receded.wheelCentre.x == untouched.wheelCentre.x);
        REQUIRE(receded.wheelCentre.y == untouched.wheelCentre.y);
        REQUIRE(receded.camber == untouched.camber);
        REQUIRE(receded.toe == untouched.toe);
        REQUIRE(receded.halfTrack == untouched.halfTrack);
    }
}

TEST_CASE("the recession map is linear inside the band's context and saturates outside it",
          "[physics][suspension][compliance]")
{
    // The guard, pinned. Inside the band's own context the map IS the coefficient times the force
    // — the sourced linearity is not distorted anywhere it is claimed. Outside it — a kerb
    // strike's one-tick ±20 kN spike, measured at ±222 mm of hub displacement unguarded on the
    // scripted launch — it saturates at ±50 mm, the band's in-context edge, a stated bound on an
    // extrapolation and not a modelled bump stop.
    constexpr auto sixPerKilonewton = 6.0 * 1.0e-3 / 1000.0;
    constexpr auto tenPerKilonewton = 10.0 * 1.0e-3 / 1000.0;

    // The linear cases are Approx because the product is arithmetic on converted units; the
    // saturated ones are exact because the map returns its own literal.
    REQUIRE(raceengine::recessionDisplacement(sixPerKilonewton, -3000.0) == Catch::Approx(-0.018).epsilon(1e-12));
    REQUIRE(raceengine::recessionDisplacement(sixPerKilonewton, 3000.0) == Catch::Approx(0.018).epsilon(1e-12));
    STATIC_REQUIRE(raceengine::recessionDisplacement(tenPerKilonewton, -22269.0) == -0.05);
    STATIC_REQUIRE(raceengine::recessionDisplacement(tenPerKilonewton, 22269.0) == 0.05);
    STATIC_REQUIRE(raceengine::recessionDisplacement(0.0, 22269.0) == 0.0);
}

TEST_CASE("the front bump travel is Dominic's measurement, ride height to the stop fully crushed",
          "[physics][suspension][golf]")
{
    // 67-73 mm of wheel rise from static ride to the hard end of travel, measured by Dominic on
    // the real car with the convention resolved 2026-08-29 (ride height to the stop fully
    // crushed). The clamp is that hard end, so the solved travel at `bumpAngle` must sit inside
    // his band — asserted against the linkage rather than as a copy of the angle, so a hardpoint
    // change that silently moves the travel fails here even with the angle untouched.
    for (const auto side : {CornerSide::Left, CornerSide::Right})
    {
        const auto hardpoints = golfMk7FrontCorner(side);
        const auto atClamp = solvedAt(hardpoints, hardpoints.bumpAngle);

        CAPTURE(outboardSign(side), atClamp.wheelTravel);

        REQUIRE(atClamp.wheelTravel > 0.067);
        REQUIRE(atClamp.wheelTravel < 0.073);
    }
}

TEST_CASE("which axle states recession and what it states", "[physics][suspension][compliance]")
{
    // Pinned at zero on all four corners, and the pin is the point: the only published figures for
    // this channel are Heissing/Ersoy's design targets (front 4-8 mm/kN of braking force; rear
    // 8-16 mm PER G — a different unit), and a design target does not set a car number. The day a
    // measurement is sourced — or the day a target is stated on Dominic's word — this fails and the
    // entry that flips it has to say which grade of number it stated. `front.recession` /
    // `rear.recession` on the sheet are the A/B until then.
    const auto built = golfGtiMk7();
    REQUIRE(built.has_value());

    for (auto index = std::size_t{0}; index < cornerCount; index++)
    {
        REQUIRE(built->corners[index].longitudinalForceRecession == 0.0);
    }
}
