#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine;

using raceengine::CarEvent;
using raceengine::carEventName;
using raceengine::deriveCarAudio;
using raceengine::DrivelineState;
using raceengine::DrivelineTorques;
using raceengine::parseSoundBankMap;
using raceengine::parseSoundGuid;
using raceengine::placeholderDriveline;
using raceengine::radiansPerSecondToRpm;
using raceengine::ShiftPhase;
using raceengine::VehicleInput;
using raceengine::VehicleState;
using raceengine::VehicleStep;

namespace
{

// The GUID map an Assetto Corsa car ships, trimmed to the lines that matter and keeping one of each
// kind that has to be read past.
constexpr std::string_view golfGuids = R"({47dcf37a-6a8d-4ef6-9f82-2e0398fa69f9} bank:/common
{026643c1-7a2f-486c-983e-52b241bf4a19} bank:/vw_golf_gti_mk7.5
{08f6ba78-f576-4f9f-a232-47657d5fec19} bus:/
{5206e42b-ae8f-49f6-b059-d613b2947b49} bus:/grp_engine_ext
{ad87ebcc-27a5-4cf9-8bf1-b7ace90156b0} event:/cars/vw_golf_gti_mk7.5/engine_ext
{22821cdc-9832-44ad-98e9-ca3212085353} event:/cars/vw_golf_gti_mk7.5/engine_int
{bdad6001-12d2-4c58-8737-86270a646ae2} event:/cars/vw_golf_gti_mk7.5/limiter
{abedfc0c-6c5e-4db2-87b8-a9bf623c8d97} event:/cars/vw_golf_gti_mk7.5/horn
{94f04703-a92c-49cf-a097-734f3e79ec08} vca:/vca_engine_ext
)";

} // namespace

TEST_CASE("a GUID reads the way FMOD writes it", "[audio][bank]")
{
    // Microsoft's layout: the first three fields are numbers and the last eight are bytes in the
    // order written. Reading the fourth group as a number reverses two of them, which produces a
    // perfectly well-formed GUID that addresses nothing — and an event that does not resolve is
    // silent rather than an error.
    const auto guid = parseSoundGuid("{ad87ebcc-27a5-4cf9-8bf1-b7ace90156b0}");
    REQUIRE(guid.has_value());

    REQUIRE(guid->data1 == 0xad87ebccu);
    REQUIRE(guid->data2 == 0x27a5u);
    REQUIRE(guid->data3 == 0x4cf9u);
    REQUIRE(guid->data4 == std::array<std::uint8_t, 8>{0x8b, 0xf1, 0xb7, 0xac, 0xe9, 0x01, 0x56, 0xb0});

    // With or without the braces, because the file has them and a caller pasting one may not.
    const auto bare = parseSoundGuid("ad87ebcc-27a5-4cf9-8bf1-b7ace90156b0");
    REQUIRE(bare.has_value());
    REQUIRE(bare->data1 == guid->data1);
    REQUIRE(bare->data4 == guid->data4);
}

TEST_CASE("a GUID that is not one is refused", "[audio][bank]")
{
    REQUIRE_FALSE(parseSoundGuid("").has_value());
    REQUIRE_FALSE(parseSoundGuid("{}").has_value());
    REQUIRE_FALSE(parseSoundGuid("{ad87ebcc-27a5-4cf9-8bf1-b7ace90156}").has_value());
    REQUIRE_FALSE(parseSoundGuid("{zd87ebcc-27a5-4cf9-8bf1-b7ace90156b0}").has_value());
    // Right length, wrong shape: the dashes are where the format says or it is not a GUID.
    REQUIRE_FALSE(parseSoundGuid("{ad87ebcc027a5-4cf9-8bf1-b7ace90156b0").has_value());
}

TEST_CASE("a car's GUID map names its events by role", "[audio][bank]")
{
    const auto map = parseSoundBankMap(golfGuids, "vw_golf_gti_mk7.5.bank");
    REQUIRE(map.has_value());

    SECTION("the events it carries")
    {
        REQUIRE(map->has(CarEvent::EngineExterior));
        REQUIRE(map->has(CarEvent::EngineInterior));
        REQUIRE(map->has(CarEvent::Limiter));
        REQUIRE(map->has(CarEvent::Horn));

        REQUIRE(map->event(CarEvent::EngineExterior).data1 == 0xad87ebccu);
    }

    SECTION("and the ones it does not, which is a fact rather than a failure")
    {
        // A bank without a turbo event is a naturally aspirated car, not a broken file.
        REQUIRE_FALSE(map->has(CarEvent::Turbo));
        REQUIRE_FALSE(map->has(CarEvent::Bodywork));
    }

    SECTION("the bank line is the car's own, not the shared one")
    {
        // `bank:/common` is AC's shared bank and comes first in every one of these files. Taking it
        // would load the wrong bank and resolve none of the car's events.
        REQUIRE(map->bank.data1 == 0x026643c1u);
    }

    SECTION("buses and VCAs are read past rather than mistaken for events")
    {
        // `bus:/grp_engine_ext` ends in the same word as `event:/.../engine_ext`, so a parser keying
        // on the last path element alone would take the bus and address the mixer instead.
        REQUIRE(map->event(CarEvent::EngineExterior).data1 != 0x5206e42bu);
    }
}

TEST_CASE("a GUID map with nothing to load says so", "[audio][bank]")
{
    // Every event and no bank line: nothing states which file to open.
    const auto orphan =
        parseSoundBankMap("{ad87ebcc-27a5-4cf9-8bf1-b7ace90156b0} event:/cars/x/engine_ext\n", "x.bank");
    REQUIRE_FALSE(orphan.has_value());
    REQUIRE(orphan.error().contains("bank:"));
}

TEST_CASE("engine speed reaches the bank in rpm", "[audio][car]")
{
    // The conversion this file exists to hold. `Engine RPM [rpm]` carried rad/s for the whole of
    // milestone one and every engine figure the project produced read 9.55x low — it survived because
    // a uniform scale error preserves every shape a test checks, and only an absolute comparison
    // catches it. This is that comparison.
    const auto setup = placeholderDriveline();

    auto driveline = DrivelineState{};
    driveline.engineSpeed = 314.159265358979; // 3000 rpm

    const auto state =
        deriveCarAudio(setup, driveline, DrivelineTorques{}, VehicleState{}, VehicleStep{}, VehicleInput{});

    REQUIRE(state.engineRpm == Catch::Approx(3000.0).epsilon(1e-6));
    REQUIRE(radiansPerSecondToRpm == Catch::Approx(9.5492965855));
}

TEST_CASE("a stalled engine is silent rather than negative", "[audio][car]")
{
    const auto setup = placeholderDriveline();

    auto driveline = DrivelineState{};
    driveline.engineSpeed = -5.0;

    const auto state =
        deriveCarAudio(setup, driveline, DrivelineTorques{}, VehicleState{}, VehicleStep{}, VehicleInput{});

    REQUIRE(state.engineRpm == 0.0);
}

TEST_CASE("load is what the engine is delivering, not what the pedal asks for", "[audio][car]")
{
    // The difference a designer means by load. A wide-open throttle against a stalled torque
    // converter is full pedal and almost no load, and it sounds like it — so this is taken from the
    // torque the coupling actually passed.
    const auto setup = placeholderDriveline();

    auto driveline = DrivelineState{};
    driveline.engineSpeed = 300.0;

    auto input = VehicleInput{};
    input.throttle = 1.0;

    const auto available = setup.engine.torque.at(300.0);
    REQUIRE(available > 0.0);

    auto pulling = DrivelineTorques{};
    pulling.clutch = available;

    auto slipping = DrivelineTorques{};
    slipping.clutch = 0.1 * available;

    const auto hard = deriveCarAudio(setup, driveline, pulling, VehicleState{}, VehicleStep{}, input);
    const auto light = deriveCarAudio(setup, driveline, slipping, VehicleState{}, VehicleStep{}, input);

    REQUIRE(hard.throttle == 1.0);
    REQUIRE(light.throttle == 1.0);

    REQUIRE(hard.load == Catch::Approx(1.0));
    REQUIRE(light.load == Catch::Approx(0.1));
}

TEST_CASE("the skid follows the worst wheel rather than the average", "[audio][car]")
{
    // A bank has one skid event and what a driver hears is whichever tyre is losing. Averaging four
    // would make a locked front under braking quieter than the same car sliding gently on all four,
    // which is the opposite of true.
    const auto setup = placeholderDriveline();

    auto vehicle = VehicleState{};
    vehicle.chassis.mass = 1400.0;

    auto step = VehicleStep{};
    for (auto& corner : step.corners)
    {
        corner.contact.slip.slipRatio = 0.02;
        corner.forces.tireVertical = 3400.0;
    }

    const auto gripping = deriveCarAudio(setup, DrivelineState{}, DrivelineTorques{}, vehicle, step, VehicleInput{});
    REQUIRE(gripping.wheelSlip < 0.05);

    step.corners[2].contact.slip.slipRatio = 0.9;

    const auto sliding = deriveCarAudio(setup, DrivelineState{}, DrivelineTorques{}, vehicle, step, VehicleInput{});
    REQUIRE(sliding.wheelSlip == Catch::Approx(0.9).epsilon(0.02));

    // And the load under that wheel, so a wheel in the air is quiet however fast it is spinning.
    step.corners[2].forces.tireVertical = 0.0;

    const auto airborne = deriveCarAudio(setup, DrivelineState{}, DrivelineTorques{}, vehicle, step, VehicleInput{});
    REQUIRE(airborne.wheelSlip == Catch::Approx(0.9).epsilon(0.02));
    REQUIRE(airborne.slipLoad == 0.0);
}

TEST_CASE("the limiter and a shift are states rather than events", "[audio][car]")
{
    // Both come and go at the car's own rate, so a bank crossfades a layer against them. Triggering a
    // one-shot instead would stutter at exactly the rate the limiter cycles — measured at 40 a second
    // on this engine.
    const auto setup = placeholderDriveline();

    auto driveline = DrivelineState{};
    driveline.engineSpeed = 700.0;
    driveline.fuelCut = true;
    driveline.shiftPhase = ShiftPhase::Neutral;

    const auto state =
        deriveCarAudio(setup, driveline, DrivelineTorques{}, VehicleState{}, VehicleStep{}, VehicleInput{});

    REQUIRE(state.onLimiter);
    REQUIRE(state.shifting);
}

TEST_CASE("nothing that is not a number reaches the bank", "[audio][car]")
{
    // FMOD takes a float and does not check it. A NaN parameter is a silent event or a stuck one
    // depending on the version, and either way it is unrecoverable without a restart — so it is
    // stopped here, where there is still something to say about it.
    const auto setup = placeholderDriveline();
    const auto nan = std::numeric_limits<double>::quiet_NaN();

    auto input = VehicleInput{};
    input.throttle = nan;

    auto torques = DrivelineTorques{};
    torques.clutch = nan;
    torques.clutchSlip = nan;

    auto vehicle = VehicleState{};
    vehicle.chassis.mass = 1400.0;

    auto step = VehicleStep{};
    step.corners[0].contact.slip.slipRatio = nan;

    const auto state = deriveCarAudio(setup, DrivelineState{}, torques, vehicle, step, input);

    REQUIRE(std::isfinite(state.engineRpm));
    REQUIRE(std::isfinite(state.throttle));
    REQUIRE(std::isfinite(state.load));
    REQUIRE(std::isfinite(state.clutchSlip));
    REQUIRE(std::isfinite(state.wheelSlip));
    REQUIRE(std::isfinite(state.slipLoad));
    REQUIRE(std::isfinite(state.roadSpeed));
}

TEST_CASE("every event this engine knows has a name", "[audio][bank]")
{
    // The name is what the GUID map is keyed on, so an event added to the enum without one silently
    // stops matching and the car loses that sound with nothing said.
    for (auto index = std::size_t{0}; index < static_cast<std::size_t>(CarEvent::Count); index++)
    {
        REQUIRE_FALSE(carEventName(static_cast<CarEvent>(index)).empty());
    }
}

TEST_CASE("rolling speed is the tread's and not the car's", "[audio][car]")
{
    // A locked wheel slides at road speed with its tread stopped, and a burnout is the reverse —
    // rolling noise driven from the chassis would play tread hiss from a tyre that is not turning.
    const auto setup = placeholderDriveline();

    auto vehicle = VehicleState{};
    vehicle.chassis.mass = 1400.0;
    vehicle.chassis.linearVelocity = glm::dvec3(0.0, 0.0, 30.0);

    auto step = VehicleStep{};
    for (auto& corner : step.corners)
    {
        corner.contact.effectiveRadius = 0.31;
    }

    for (auto& corner : vehicle.corners)
    {
        corner.wheelSpeed = 30.0 / 0.31;
    }

    const auto rolling = deriveCarAudio(setup, DrivelineState{}, DrivelineTorques{}, vehicle, step, VehicleInput{});
    REQUIRE(rolling.roadSpeed == Catch::Approx(30.0));
    REQUIRE(rolling.rollingSpeed == Catch::Approx(30.0).epsilon(0.001));

    // All four locked: the car still moves and the tread does not.
    for (auto& corner : vehicle.corners)
    {
        corner.wheelSpeed = 0.0;
    }

    const auto locked = deriveCarAudio(setup, DrivelineState{}, DrivelineTorques{}, vehicle, step, VehicleInput{});
    REQUIRE(locked.roadSpeed == Catch::Approx(30.0));
    REQUIRE(locked.rollingSpeed == 0.0);

    // One locked of four takes a quarter of the hiss, not all of it.
    vehicle.corners[0].wheelSpeed = 30.0 / 0.31;
    vehicle.corners[1].wheelSpeed = 30.0 / 0.31;
    vehicle.corners[2].wheelSpeed = 30.0 / 0.31;

    const auto threeRolling =
        deriveCarAudio(setup, DrivelineState{}, DrivelineTorques{}, vehicle, step, VehicleInput{});
    REQUIRE(threeRolling.rollingSpeed == Catch::Approx(22.5).epsilon(0.001));
}
