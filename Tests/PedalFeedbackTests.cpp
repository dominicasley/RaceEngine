// The pedal motors, stage one and stage two, with no device anywhere near them.
//
// What the cue is *for* is the one thing a steering wheel physically cannot say: which wheel has
// stopped rotating with the road, under the foot that caused it. The wheel reports the front axle's
// grip through the trail and reports nothing at all about a locked rear or about wheelspin, because
// neither reaches the steering.

#include <array>
#include <cmath>
#include <cstddef>
#include <span>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

import raceengine;

using raceengine::derivePedalFeedback;
using raceengine::mapPedalFeedback;
using raceengine::PedalFeedback;
using raceengine::PedalFeedbackSetup;
using raceengine::PedalMotorMapping;
using raceengine::PedalMotorProfile;
using raceengine::pedalWheelLimit;
using raceengine::SlippingWheel;

namespace
{

// A quarter of a 1400 kg car, which is what `minimumLoadShare` is a share of.
constexpr auto share = 3400.0;

// Four wheels gripping, front driven — the Golf's layout.
[[nodiscard]] std::array<SlippingWheel, 4> rolling()
{
    auto wheels = std::array<SlippingWheel, 4>{};

    for (auto index = std::size_t{0}; index < wheels.size(); index++)
    {
        wheels[index] = SlippingWheel{.slipRatio = 0.0,
                                      .peakSlipRatio = 0.10,
                                      .load = share,
                                      .inContact = true,
                                      .driven = index < 2};
    }

    return wheels;
}

[[nodiscard]] PedalFeedback derive(const std::array<SlippingWheel, 4>& wheels, const double brake,
                                   const double throttle)
{
    return derivePedalFeedback(PedalFeedbackSetup{}, std::span<const SlippingWheel>(wheels), brake, throttle, share);
}

} // namespace

TEST_CASE("a gripping car says nothing to either pedal", "[input][pedals]")
{
    // The case that matters most for a vibration motor and least for anything else: an eccentric
    // mass that is on when it should not be is not a subtle inaccuracy, it is a rig the driver wants
    // to unplug. Full pedal on both, four tyres gripping, and both motors silent.
    const auto answer = derive(rolling(), 1.0, 1.0);

    REQUIRE(answer.finite);
    REQUIRE(answer.brake == 0.0);
    REQUIRE(answer.throttle == 0.0);
    REQUIRE(answer.brakeWheel == pedalWheelLimit);
    REQUIRE(answer.throttleWheel == pedalWheelLimit);
}

TEST_CASE("the sign of the slip is what tells the two pedals apart", "[input][pedals]")
{
    SECTION("a wheel turning slower than the road is locking, and that is the brake's")
    {
        auto wheels = rolling();
        wheels[0].slipRatio = -0.30;

        const auto answer = derive(wheels, 1.0, 0.0);

        REQUIRE(answer.brake > 0.0);
        REQUIRE(answer.throttle == 0.0);
        REQUIRE(answer.brakeWheel == 0);
    }

    SECTION("and one turning faster is spinning, which is the throttle's")
    {
        auto wheels = rolling();
        wheels[0].slipRatio = 0.30;

        const auto answer = derive(wheels, 0.0, 1.0);

        REQUIRE(answer.throttle > 0.0);
        REQUIRE(answer.brake == 0.0);
        REQUIRE(answer.throttleWheel == 0);
    }
}

TEST_CASE("every wheel can lock but only a driven one can spin", "[input][pedals]")
{
    // **A locked rear is the one that spins the car, and it is exactly the one the steering will not
    // report** — there is no trail from an undriven, unsteered axle to the driver's hands. If the
    // brake cue watched only the front axle it would be silent for the failure a driver most needs
    // to catch.
    SECTION("a locked rear reaches the brake pedal")
    {
        auto wheels = rolling();
        wheels[3].slipRatio = -0.40;

        const auto answer = derive(wheels, 1.0, 0.0);

        REQUIRE(answer.brake > 0.0);
        REQUIRE(answer.brakeWheel == 3);
    }

    SECTION("but an undriven wheel turning faster than the road is not wheelspin")
    {
        // A rear wheel on a front-drive car cannot be driven past the road; if its ratio reads
        // positive it is a wheel being dragged over a crest or coming off a kerb, and buzzing the
        // throttle for it would be reporting an event that did not happen.
        auto wheels = rolling();
        wheels[3].slipRatio = 0.40;

        const auto answer = derive(wheels, 0.0, 1.0);

        REQUIRE(answer.throttle == 0.0);
    }
}

TEST_CASE("the cue belongs to the foot that caused it", "[input][pedals]")
{
    // Buzzing a pedal nobody is standing on is a message to an empty room, and it is also how engine
    // braking on a downshift ends up rattling the brake pedal.
    auto locking = rolling();
    locking[0].slipRatio = -0.40;

    REQUIRE(derive(locking, 0.0, 0.0).brake == 0.0);
    REQUIRE(derive(locking, 1.0, 0.0).brake > 0.0);

    auto spinning = rolling();
    spinning[0].slipRatio = 0.40;

    REQUIRE(derive(spinning, 0.0, 0.0).throttle == 0.0);
    REQUIRE(derive(spinning, 0.0, 1.0).throttle > 0.0);
}

TEST_CASE("the onset is past the tyre's own peak, not at it and not at a fixed slip", "[input][pedals]")
{
    // **At the peak the tyre is making its most force**, which is what a driver braking well is
    // sitting on. A pedal that buzzed there would buzz on every good stop and teach them to brake
    // less hard. What is worth saying is that they have gone past it.
    const auto setup = PedalFeedbackSetup{};

    auto wheels = rolling();

    const auto severityAt = [&](const double multipleOfPeak)
    {
        wheels[0].slipRatio = -multipleOfPeak * wheels[0].peakSlipRatio;

        return derive(wheels, 1.0, 0.0).brake;
    };

    REQUIRE(severityAt(1.0) == 0.0);
    REQUIRE(severityAt(setup.onsetPeaks) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(severityAt(0.5 * (setup.onsetPeaks + setup.brakeFullPeaks)) == Catch::Approx(0.5).margin(0.01));
    REQUIRE(severityAt(setup.brakeFullPeaks) == Catch::Approx(1.0));
    REQUIRE(severityAt(10.0) == Catch::Approx(1.0));

    SECTION("and it follows the compound, because it is a multiple rather than a slip")
    {
        // The same absolute slip on a tyre whose curve peaks later is *not* the same event. A cue
        // stated as an absolute slip would fire on one compound and not on another, and would drift
        // out of step with the tyre audio's skid, which is the same physical event.
        auto grippier = rolling();
        grippier[0].peakSlipRatio = 0.20;
        grippier[0].slipRatio = -0.30;

        auto ordinary = rolling();
        ordinary[0].peakSlipRatio = 0.10;
        ordinary[0].slipRatio = -0.30;

        REQUIRE(derive(grippier, 1.0, 0.0).brake < derive(ordinary, 1.0, 0.0).brake);
    }
}

TEST_CASE("the accelerator's range is wider than the brake's, because the slip is", "[input][pedals]")
{
    // **Slip ratio is bounded below at -1 and unbounded above.** A locked wheel has stopped and
    // there is nothing worse than stopped; a spinning one can read three or twenty. Measured on a
    // standing start, a front-drive hatchback reaches 24 times its peak slip — so the two cues
    // sharing one range left the accelerator's motor pinned at full for the whole launch, which
    // tells a driver that they are spinning and never how much.
    const auto setup = PedalFeedbackSetup{};
    REQUIRE(setup.throttleFullPeaks > setup.brakeFullPeaks);

    auto wheels = rolling();

    const auto at = [&](const double multipleOfPeak, const bool spinning)
    {
        wheels[0].slipRatio = (spinning ? 1.0 : -1.0) * multipleOfPeak * wheels[0].peakSlipRatio;
        const auto answer = derive(wheels, spinning ? 0.0 : 1.0, spinning ? 1.0 : 0.0);

        return spinning ? answer.throttle : answer.brake;
    };

    // The same slip is a milder message under power than under braking, which is the asymmetry.
    REQUIRE(at(2.0, true) < at(2.0, false));

    // And the accelerator still has somewhere to go where the brake has saturated.
    REQUIRE(at(setup.brakeFullPeaks, false) == Catch::Approx(1.0));
    REQUIRE(at(setup.brakeFullPeaks, true) < 0.9);
    REQUIRE(at(setup.throttleFullPeaks, true) == Catch::Approx(1.0));
}

TEST_CASE("a wheel that is barely carrying anything says nothing", "[input][pedals]")
{
    // An unloaded inside wheel on a kerb will show any slip you like and is not the reason the car
    // is not stopping. A gate rather than a weight: it is a different situation, not a quieter one.
    auto wheels = rolling();
    wheels[0].slipRatio = -0.50;

    SECTION("in the air")
    {
        wheels[0].inContact = false;
        REQUIRE(derive(wheels, 1.0, 0.0).brake == 0.0);
    }

    SECTION("or barely touching")
    {
        wheels[0].load = 0.10 * share;
        REQUIRE(derive(wheels, 1.0, 0.0).brake == 0.0);
    }

    SECTION("but carrying its share")
    {
        wheels[0].load = 0.60 * share;
        REQUIRE(derive(wheels, 1.0, 0.0).brake > 0.0);
    }
}

TEST_CASE("the worst wheel wins rather than the average", "[input][pedals]")
{
    // What a foot needs to know is that something has let go, not the mean of four things — three
    // gripping wheels must not quieten the one that is sliding.
    auto wheels = rolling();
    wheels[2].slipRatio = -0.50;

    const auto one = derive(wheels, 1.0, 0.0);

    wheels[0].slipRatio = -0.16;
    wheels[1].slipRatio = -0.14;

    const auto several = derive(wheels, 1.0, 0.0);

    REQUIRE(several.brake == Catch::Approx(one.brake));
    REQUIRE(several.brakeWheel == 2);
}

TEST_CASE("a value that is not a number stops the motors rather than reaching them", "[input][pedals]")
{
    // The worst failure this feature has: unlike a torque a vibration does not settle, and unlike a
    // sound it cannot be turned down from the desk.
    const auto nan = std::nan("");

    auto wheels = rolling();
    wheels[1].slipRatio = nan;

    const auto answer = derive(wheels, 1.0, 0.0);
    REQUIRE_FALSE(answer.finite);

    auto profile = PedalMotorProfile{};
    profile.hasMotors = true;

    const auto command = mapPedalFeedback(profile, PedalMotorMapping{}, answer);
    REQUIRE(command.silent());
}

TEST_CASE("stage two is the only thing that knows what a motor is", "[input][pedals][motors]")
{
    auto profile = PedalMotorProfile{};
    profile.hasMotors = true;

    const auto mapping = PedalMotorMapping{};

    SECTION("pedals with no motors answer silent whatever they are asked")
    {
        // CSL Elite and CSL LC pedals are the same brand behind the same driver and have none, so
        // this is the default rather than an edge case.
        const auto none = PedalMotorProfile{};
        REQUIRE_FALSE(none.hasMotors);

        REQUIRE(mapPedalFeedback(none, mapping, PedalFeedback{.brake = 1.0, .throttle = 1.0}).silent());
    }

    SECTION("nought is exactly nought, and anything above it clears the motor's stiction")
    {
        // **The number that decides whether the feature works at all.** An eccentric mass below its
        // stiction draws current and sits still, so mapping a severity of 0.05 onto 5% duty produces
        // nothing and the cue appears to have a threshold far higher than the one stage one states.
        REQUIRE(mapPedalFeedback(profile, mapping, PedalFeedback{}).silent());

        const auto faint = mapPedalFeedback(profile, mapping, PedalFeedback{.brake = 0.01});
        REQUIRE(faint.brake > 0);
        REQUIRE(static_cast<double>(faint.brake) / 255.0 >= profile.minimumDuty - 0.01);
    }

    SECTION("full severity is full scale, and the two motors are independent")
    {
        const auto both = mapPedalFeedback(profile, mapping, PedalFeedback{.brake = 1.0, .throttle = 1.0});
        REQUIRE(both.brake == 255);
        REQUIRE(both.throttle == 255);

        const auto braking = mapPedalFeedback(profile, mapping, PedalFeedback{.brake = 1.0});
        REQUIRE(braking.brake == 255);
        REQUIRE(braking.throttle == 0);
    }

    SECTION("and the word is the report the driver documents")
    {
        // Throttle in the high byte, brake in the middle: `0xFF0000` and `0xFF00` are the driver's
        // own examples, and this is that sentence written once rather than at every call site.
        REQUIRE(mapPedalFeedback(profile, mapping, PedalFeedback{.throttle = 1.0}).word() == 0xFF0000u);
        REQUIRE(mapPedalFeedback(profile, mapping, PedalFeedback{.brake = 1.0}).word() == 0x00FF00u);
        REQUIRE(mapPedalFeedback(profile, mapping, PedalFeedback{.brake = 1.0, .throttle = 1.0}).word() == 0xFFFF00u);
        REQUIRE(mapPedalFeedback(profile, mapping, PedalFeedback{}).word() == 0u);
    }

    SECTION("the driver's dial can turn one cue off without touching the other")
    {
        auto onlyBrake = PedalMotorMapping{};
        onlyBrake.throttleGain = 0.0;

        const auto command = mapPedalFeedback(profile, onlyBrake, PedalFeedback{.brake = 1.0, .throttle = 1.0});

        REQUIRE(command.brake == 255);
        REQUIRE(command.throttle == 0);
    }
}
