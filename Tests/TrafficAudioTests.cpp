#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine;

using Catch::Approx;
using raceengine::AudioListener;
using raceengine::bandCentreRpm;
using raceengine::CarAudioState;
using raceengine::classifyTrafficEngineLayers;
using raceengine::classifyTrafficTyreLayers;
using raceengine::deriveTrafficAudio;
using raceengine::EngineLayer;
using raceengine::LayerMix;
using raceengine::LayerPosition;
using raceengine::LayerThrottle;
using raceengine::mixTrafficEngineLayers;
using raceengine::noTrafficCar;
using raceengine::TrafficAudioCar;
using raceengine::TrafficAudioSettings;
using raceengine::TrafficDriveState;
using raceengine::TrafficEngineRange;
using raceengine::TrafficGearbox;
using raceengine::TrafficVoice;
using raceengine::TrafficVoiceAllocator;
using raceengine::TyreNoise;

// The fleet's banks as the exporter writes them out (`docs/traffic-system-brief.md`, the fleet), by
// the sample names inside the bank. Five cars, four naming conventions, and none of them the Golf's.

namespace
{

[[nodiscard]] std::vector<std::string> names(std::initializer_list<const char*> list)
{
    return std::vector<std::string>(list.begin(), list.end());
}

[[nodiscard]] std::size_t countWhere(const std::vector<EngineLayer>& layers, const LayerPosition position,
                                     const LayerThrottle throttle)
{
    return static_cast<std::size_t>(std::count_if(layers.begin(), layers.end(),
                                                  [&](const EngineLayer& layer) {
                                                      return layer.position == position &&
                                                             layer.throttle == throttle;
                                                  }));
}

[[nodiscard]] const EngineLayer* find(const std::vector<EngineLayer>& layers, const std::string& name)
{
    const auto found = std::find_if(layers.begin(), layers.end(),
                                    [&name](const EngineLayer& layer) { return layer.name == name; });

    return found == layers.end() ? nullptr : &*found;
}

[[nodiscard]] double gainSum(const std::vector<LayerMix>& mix)
{
    auto sum = 0.0;
    for (const auto& entry : mix)
    {
        sum += entry.gain;
    }

    return sum;
}

// The Commodore's bank: two rpm-named families, the exhaust recorded outside and the engine inside,
// with the bangs, pops, horn and doors between them.
const auto commodore = names({"7 XBang05",       "13 ExhL_07540", "5 EngA_04364",  "2 EngA_02705",
                              "6 XBang04",       "6 EngA_04904",  "1 EngA_02165",  "3 ExhL_01619",
                              "12 ExhL_07007",   "23 Asph_HDF_SkidHigh_3V4", "traction_control",
                              "gearup",          "9 EngA_06549",  "geardnEXT",     "7 ExhL_04315",
                              "2 ExhL_01079",    "3 XBang01",     "gearupEXT",     "10 EngA_07264",
                              "10 ExhL_05925",   "4 ExhL_02155",  "5 ExhL_02695",  "6 ExhL_03773",
                              "2 pop2",          "5 XBang03",     "8 ExhL_04850",  "4 EngA_03829",
                              "3 EngA_03288",    "1 pop1",        "bodywork",      "door_open",
                              "1 ExhL_00826",    "7 EngA_05477",  "11 ExhL_06465", "transmission",
                              "8 EngA_06032",    "1 HornA_Long_01", "9 ExhL_05389", "door_close",
                              "4 XBang02"});

} // namespace

TEST_CASE("an rpm-named bank keeps its largest exterior family and drops the rest", "[audio][traffic-audio]")
{
    const auto layers = classifyTrafficEngineLayers(commodore, 640.0, 6600.0);

    // Thirteen `ExhL` recordings, every one of them the car on the throttle heard from outside; the
    // ten `EngA` recordings are the same instants from the engine bay and are not opened.
    REQUIRE(layers.size() == 13);
    CHECK(countWhere(layers, LayerPosition::Exterior, LayerThrottle::On) == 13);

    for (const auto& layer : layers)
    {
        CHECK(layer.name.find("ExhL") != std::string::npos);
    }

    // The rpm is the recording's own, not a band's centre.
    const auto* lowest = find(layers, "1 ExhL_00826");
    const auto* highest = find(layers, "13 ExhL_07540");
    REQUIRE(lowest != nullptr);
    REQUIRE(highest != nullptr);
    CHECK(lowest->centreRpm == Approx(826.0));
    CHECK(highest->centreRpm == Approx(7540.0));

    // The ordinal in front is not a speed, and the skid sample's `3V4` is not one either.
    CHECK(find(layers, "23 Asph_HDF_SkidHigh_3V4") == nullptr);
    CHECK(find(layers, "1 HornA_Long_01") == nullptr);

    // No tread loop in this bank.
    CHECK(classifyTrafficTyreLayers(commodore).empty());
}

TEST_CASE("three families of the same instants keep the exterior one whatever its count", "[audio][traffic-audio]")
{
    // The Carnival: `EngA` and `EngB` are two microphones on the engine, `Exh0` the tailpipe. The
    // interior families outnumber the exterior one two to one and are still the ones dropped.
    const auto carnival = names({"5 EngA_02808", "8 EngA_04328", "7 EngB_03687", "8 Exh0_04328", "5 EngB_02808",
                                 "1 EngB_00852", "7 Exh0_03687", "9 Exh0_04963", "3 Exh0_01835", "tyre_rolling",
                                 "Skid_street", "Skidloud", "veh_fi_turbo_bov_McLaren650s_0_5_0", "8 HornC_Long_01",
                                 "12 IntakeChuff1_6A1"});

    const auto layers = classifyTrafficEngineLayers(carnival, 800.0, 4700.0);

    REQUIRE(layers.size() == 4);
    for (const auto& layer : layers)
    {
        CHECK(layer.position == LayerPosition::Exterior);
        CHECK(layer.name.find("Exh0") != std::string::npos);
    }

    // The turbo sample's `0_5_0` suffix is three short numbers and not a speed.
    CHECK(find(layers, "veh_fi_turbo_bov_McLaren650s_0_5_0") == nullptr);

    // The tread loop classifies; the two skids are left closed.
    const auto tyres = classifyTrafficTyreLayers(carnival);
    REQUIRE(tyres.size() == 1);
    CHECK(tyres.front().name == "tyre_rolling");
    CHECK(tyres.front().noise == TyreNoise::Rolling);
}

TEST_CASE("fused state-and-band names read as the Golf's separated ones do", "[audio][traffic-audio]")
{
    // The Legacy: `GC8_ex_in_onhigh` is `ex` + `on` + `high` with the last two run together, and the
    // first position word wins so `ex_in` is the outside. Takes on the end are dropped.
    const auto legacy = names({"GC8_ex_in_onhigh", "GC8_ex_in_offverylow", "GC8_ex_in_onverylow4",
                               "GC8_ex_in_onverylow2", "GC8_in_on_low2", "GC8_hood_verylow4", "GC8_ex_in_rev",
                               "GC8_in_idle", "GC8_ex_in_onidle", "GC8_ex_acc1__2_", "GC8_in_on_lowmid"});

    const auto layers = classifyTrafficEngineLayers(legacy, 850.0, 6600.0);

    REQUIRE(layers.size() == 6);
    CHECK(countWhere(layers, LayerPosition::Exterior, LayerThrottle::On) == 3);
    CHECK(countWhere(layers, LayerPosition::Exterior, LayerThrottle::Off) == 1);
    CHECK(countWhere(layers, LayerPosition::Interior, LayerThrottle::On) == 1);
    CHECK(countWhere(layers, LayerPosition::Interior, LayerThrottle::Idle) == 1);

    const auto* high = find(layers, "GC8_ex_in_onhigh");
    REQUIRE(high != nullptr);
    CHECK(high->position == LayerPosition::Exterior);
    CHECK(high->throttle == LayerThrottle::On);
    CHECK(high->centreRpm == Approx(bandCentreRpm(4, 850.0, 6600.0)));

    const auto* overrun = find(layers, "GC8_ex_in_offverylow");
    REQUIRE(overrun != nullptr);
    CHECK(overrun->throttle == LayerThrottle::Off);
    CHECK(overrun->centreRpm == Approx(bandCentreRpm(1, 850.0, 6600.0)));

    // `onidle` is the engine held against the throttle at idle speed, not the idle state.
    const auto* heldAtIdle = find(layers, "GC8_ex_in_onidle");
    REQUIRE(heldAtIdle != nullptr);
    CHECK(heldAtIdle->throttle == LayerThrottle::On);
    CHECK(heldAtIdle->centreRpm == Approx(850.0));

    // One of the two `onverylow` takes, and it is the first listed.
    CHECK(find(layers, "GC8_ex_in_onverylow4") != nullptr);
    CHECK(find(layers, "GC8_ex_in_onverylow2") == nullptr);

    // A band with no throttle word, a state with no band, a sweep, and a band this engine does not
    // name are all left out rather than guessed at.
    CHECK(find(layers, "GC8_hood_verylow4") == nullptr);
    CHECK(find(layers, "GC8_ex_in_rev") == nullptr);
    CHECK(find(layers, "GC8_ex_acc1__2_") == nullptr);
    CHECK(find(layers, "GC8_in_on_lowmid") == nullptr);
}

TEST_CASE("numeric bands and an ext prefix classify, and a bare number does not", "[audio][traffic-audio]")
{
    // The Hilux carries a Fiat 500's bank: `ext500ss_on_7500` marks the outside as a prefix on the
    // engine's own name, `4279_off` is a speed and a state with nothing else, and `7025` is a sample
    // the engine has no use for.
    const auto hilux = names({"ext500ss_on_7500", "500ss_idle", "ext500ss_idle", "4279_off", "7025", "3835a_off",
                              "500_limiter", "ext500ss_off_2500", "backfireEXT_3", "500ss_on_7500", "geardnEXT"});

    const auto layers = classifyTrafficEngineLayers(hilux, 850.0, 4300.0);

    REQUIRE(layers.size() == 6);
    CHECK(countWhere(layers, LayerPosition::Exterior, LayerThrottle::On) == 1);
    CHECK(countWhere(layers, LayerPosition::Exterior, LayerThrottle::Off) == 1);
    CHECK(countWhere(layers, LayerPosition::Exterior, LayerThrottle::Idle) == 1);
    CHECK(countWhere(layers, LayerPosition::Interior, LayerThrottle::On) == 1);
    CHECK(countWhere(layers, LayerPosition::Interior, LayerThrottle::Off) == 1);
    CHECK(countWhere(layers, LayerPosition::Interior, LayerThrottle::Idle) == 1);

    const auto* outsideOn = find(layers, "ext500ss_on_7500");
    REQUIRE(outsideOn != nullptr);
    CHECK(outsideOn->position == LayerPosition::Exterior);
    CHECK(outsideOn->centreRpm == Approx(7500.0));

    const auto* outsideIdle = find(layers, "ext500ss_idle");
    REQUIRE(outsideIdle != nullptr);
    CHECK(outsideIdle->throttle == LayerThrottle::Idle);
    CHECK(outsideIdle->centreRpm == Approx(850.0));

    const auto* numericOff = find(layers, "4279_off");
    REQUIRE(numericOff != nullptr);
    CHECK(numericOff->position == LayerPosition::Interior);
    CHECK(numericOff->throttle == LayerThrottle::Off);
    CHECK(numericOff->centreRpm == Approx(4279.0));

    CHECK(find(layers, "7025") == nullptr);
    CHECK(find(layers, "3835a_off") == nullptr);
    CHECK(find(layers, "500_limiter") == nullptr);
    CHECK(find(layers, "backfireEXT_3") == nullptr);
}

TEST_CASE("a bank recorded only from the driver's seat is promoted to the pavement", "[audio][traffic-audio]")
{
    const auto interiorOnly = names({"4c_in_on_low", "4c_in_off_low", "4c_in_idle"});

    const auto layers = classifyTrafficEngineLayers(interiorOnly, 800.0, 6000.0);

    REQUIRE(layers.size() == 3);
    for (const auto& layer : layers)
    {
        CHECK(layer.position == LayerPosition::Exterior);
    }
}

TEST_CASE("the traffic tyre classifier keeps the tread loop alone", "[audio][traffic-audio]")
{
    const auto tyres = classifyTrafficTyreLayers(names({"tyre_rolling", "Skid", "skid_ext_mono", "flat_tyre_mono"}));

    REQUIRE(tyres.size() == 1);
    CHECK(tyres.front().name == "tyre_rolling");
    CHECK(tyres.front().noise == TyreNoise::Rolling);
}

TEST_CASE("a bank with no idle recording has its lowest layer carry the idle, pitched down", "[audio][traffic-audio]")
{
    const auto layers = classifyTrafficEngineLayers(names({"1 ExhL_00826", "5 ExhL_02695", "13 ExhL_07540"}), 640.0,
                                                    6600.0);
    REQUIRE(layers.size() == 3);

    auto gearbox = TrafficGearbox{};
    auto drive = TrafficDriveState{};
    const auto idling = deriveTrafficAudio(gearbox, drive, 0.0, 0.0, 640.0, 6600.0);
    REQUIRE(idling.engineRpm == Approx(640.0));

    auto mix = std::vector<LayerMix>(layers.size());
    mixTrafficEngineLayers(layers, idling, 640.0, mix);

    // One loop, the lowest, at its full level and slowed to the idle speed.
    const auto* lowest = find(layers, "1 ExhL_00826");
    REQUIRE(lowest != nullptr);
    const auto index = static_cast<std::size_t>(lowest - layers.data());

    CHECK(mix[index].gain == Approx(1.0));
    CHECK(mix[index].pitch == Approx(640.0 / 826.0));
    CHECK(gainSum(mix) == Approx(1.0));
}

TEST_CASE("a bank with no overrun recordings carries the whole load on its throttle set", "[audio][traffic-audio]")
{
    const auto layers = classifyTrafficEngineLayers(names({"1 ExhL_00826", "5 ExhL_02695", "13 ExhL_07540"}), 640.0,
                                                    6600.0);

    // A car holding its speed asks for a third of the load; with nothing recorded off the throttle
    // that third would be two thirds of silence.
    auto cruising = CarAudioState{};
    cruising.engineRpm = 2695.0;
    cruising.idleRpm = 640.0;
    cruising.limiterRpm = 6600.0;
    cruising.load = 0.35;
    cruising.throttle = 0.35;

    auto mix = std::vector<LayerMix>(layers.size());
    mixTrafficEngineLayers(layers, cruising, 640.0, mix);

    const auto* middle = find(layers, "5 ExhL_02695");
    REQUIRE(middle != nullptr);
    CHECK(mix[static_cast<std::size_t>(middle - layers.data())].gain == Approx(1.0));
    CHECK(gainSum(mix) == Approx(1.0));
}

TEST_CASE("the mix's allowances are judged over the exterior layers alone", "[audio][traffic-audio]")
{
    // The Charger's shape: an exterior set on the throttle only, and overrun loops recorded inside.
    // Counting the interior overrun would hold the load at the driver's fraction and hand the rest
    // to a set the street cannot hear.
    const auto layers = classifyTrafficEngineLayers(names({"x_ex_on_low", "x_ex_on_high", "x_in_off_low"}), 800.0,
                                                    6000.0);
    REQUIRE(layers.size() == 3);

    auto cruising = CarAudioState{};
    cruising.engineRpm = bandCentreRpm(2, 800.0, 6000.0);
    cruising.idleRpm = 800.0;
    cruising.limiterRpm = 6000.0;
    cruising.load = 0.35;
    cruising.throttle = 0.35;

    auto mix = std::vector<LayerMix>(layers.size());
    mixTrafficEngineLayers(layers, cruising, 800.0, mix);

    const auto* low = find(layers, "x_ex_on_low");
    REQUIRE(low != nullptr);
    CHECK(mix[static_cast<std::size_t>(low - layers.data())].gain == Approx(1.0));

    const auto* insideOff = find(layers, "x_in_off_low");
    REQUIRE(insideOff != nullptr);
    CHECK(mix[static_cast<std::size_t>(insideOff - layers.data())].gain == 0.0);

    // And the mirror: an idle recorded inside only is not an idle the street has, so the lowest
    // exterior loop carries it rather than the idle state taking the weight to nothing.
    const auto withInsideIdle =
        classifyTrafficEngineLayers(names({"x_ex_on_low", "x_ex_off_low", "x_in_idle"}), 800.0, 6000.0);
    REQUIRE(withInsideIdle.size() == 3);

    auto idling = CarAudioState{};
    idling.engineRpm = 800.0;
    idling.idleRpm = 800.0;
    idling.limiterRpm = 6000.0;
    idling.load = 0.35;

    auto idleMix = std::vector<LayerMix>(withInsideIdle.size());
    mixTrafficEngineLayers(withInsideIdle, idling, 800.0, idleMix);

    CHECK(gainSum(idleMix) > 0.9);
    const auto* inside = find(withInsideIdle, "x_in_idle");
    REQUIRE(inside != nullptr);
    CHECK(idleMix[static_cast<std::size_t>(inside - withInsideIdle.data())].gain == 0.0);
}

TEST_CASE("a bank with an idle recording plays it at idle", "[audio][traffic-audio]")
{
    const auto layers = classifyTrafficEngineLayers(names({"x_ex_idle", "x_ex_on_low", "x_ex_off_low"}), 800.0, 6000.0);
    REQUIRE(layers.size() == 3);

    auto idling = CarAudioState{};
    idling.engineRpm = 800.0;
    idling.idleRpm = 800.0;
    idling.limiterRpm = 6000.0;
    idling.load = 0.5;

    auto mix = std::vector<LayerMix>(layers.size());
    mixTrafficEngineLayers(layers, idling, 800.0, mix);

    const auto* idle = find(layers, "x_ex_idle");
    REQUIRE(idle != nullptr);
    CHECK(mix[static_cast<std::size_t>(idle - layers.data())].gain == Approx(1.0));
    CHECK(gainSum(mix) == Approx(1.0));
}

TEST_CASE("the virtual gearbox idles at rest, never passes the limiter and shifts up through the range",
          "[audio][traffic-audio]")
{
    const auto gearbox = TrafficGearbox{};
    auto drive = TrafficDriveState{};

    const auto rest = deriveTrafficAudio(gearbox, drive, 0.0, 0.0, 800.0, 6500.0);
    CHECK(rest.engineRpm == Approx(800.0));
    CHECK(rest.gear == 1);
    CHECK(rest.load == Approx(0.35));
    CHECK(rest.idleRpm == Approx(800.0));
    CHECK(rest.limiterRpm == Approx(6500.0));

    auto lastGear = std::int32_t{0};
    for (auto speed = 0.0; speed <= 60.0; speed += 0.25)
    {
        const auto state = deriveTrafficAudio(gearbox, drive, speed, 0.0, 800.0, 6500.0);

        CHECK(state.engineRpm >= 800.0 - 1e-9);
        CHECK(state.engineRpm <= 6500.0 + 1e-9);
        CHECK(state.gear >= lastGear);
        CHECK(state.roadSpeed == Approx(speed));
        CHECK(state.rollingSpeed == Approx(speed));
        CHECK(state.wheelSlip == 0.0);

        lastGear = state.gear;
    }

    CHECK(lastGear == 5);

    // Top gear at an impossible speed is held on the limiter rather than past it.
    const auto flatOut = deriveTrafficAudio(gearbox, drive, 100.0, 0.0, 800.0, 6500.0);
    CHECK(flatOut.engineRpm == Approx(6500.0));
    CHECK(flatOut.gear == 5);
}

TEST_CASE("the virtual gearbox has hysteresis and holds a gear longer under hard acceleration",
          "[audio][traffic-audio]")
{
    const auto gearbox = TrafficGearbox{};

    // First gear runs out at 0.42 of the limiter: 2730 rpm, 7.18 m/s. A car that came down to 6.5
    // from above stays in second, where a car that arrived at 6.5 from rest is in first.
    auto fromRest = TrafficDriveState{};
    const auto arriving = deriveTrafficAudio(gearbox, fromRest, 6.5, 0.0, 800.0, 6500.0);
    CHECK(arriving.gear == 1);

    auto fromAbove = TrafficDriveState{};
    const auto passing = deriveTrafficAudio(gearbox, fromAbove, 7.3, 0.0, 800.0, 6500.0);
    CHECK(passing.gear == 2);
    const auto slowing = deriveTrafficAudio(gearbox, fromAbove, 6.5, 0.0, 800.0, 6500.0);
    CHECK(slowing.gear == 2);
    CHECK(slowing.engineRpm < arriving.engineRpm);

    // And well below the down-shift fraction it comes back to first.
    const auto crawling = deriveTrafficAudio(gearbox, fromAbove, 3.0, 0.0, 800.0, 6500.0);
    CHECK(crawling.gear == 1);

    // Nine metres a second is second gear for a driver holding speed and still first for one
    // pulling 2.5 m/s^2, whose note is the higher for it.
    auto coasting = TrafficDriveState{};
    auto eager = TrafficDriveState{};
    const auto held = deriveTrafficAudio(gearbox, coasting, 9.0, 0.0, 800.0, 6500.0);
    const auto pulling = deriveTrafficAudio(gearbox, eager, 9.0, 2.5, 800.0, 6500.0);
    CHECK(held.gear == 2);
    CHECK(pulling.gear == 1);
    CHECK(pulling.engineRpm > held.engineRpm);

    // The load reads the acceleration around the light load of holding a speed.
    CHECK(held.load == Approx(0.35));
    CHECK(pulling.load == Approx(1.0));
    auto braking = TrafficDriveState{};
    CHECK(deriveTrafficAudio(gearbox, braking, 9.0, -1.0, 800.0, 6500.0).load == Approx(0.0));

    // Nothing a lane model can produce takes it out of range.
    auto broken = TrafficDriveState{};
    const auto nan = std::nan("");
    const auto safe = deriveTrafficAudio(gearbox, broken, nan, nan, 800.0, 6500.0);
    CHECK(safe.engineRpm == Approx(800.0));
    CHECK(std::isfinite(safe.load));
}

namespace
{

[[nodiscard]] TrafficAudioCar carAt(const std::uint32_t id, const double zMetres, const std::uint8_t body = 0,
                                     const double speed = 0.0)
{
    return TrafficAudioCar{.id = id,
                           .body = body,
                           .positionMetres = glm::dvec3(0.0, 0.0, zMetres),
                           .velocityMetresPerSecond = glm::dvec3(0.0, 0.0, speed),
                           .accelerationMetresPerSecondSquared = 0.0};
}

[[nodiscard]] TrafficAudioSettings twoVoices()
{
    auto settings = TrafficAudioSettings{};
    settings.voices = 2;
    settings.acquireMetres = 100.0;
    settings.releaseMetres = 140.0;
    settings.preemptRatio = 0.7;
    settings.fadeSeconds = 0.1;

    return settings;
}

[[nodiscard]] const TrafficVoice* voiceFor(const TrafficVoiceAllocator& allocator, const std::uint32_t car)
{
    for (const auto& voice : allocator.voices())
    {
        if (voice.live && voice.car == car)
        {
            return &voice;
        }
    }

    return nullptr;
}

[[nodiscard]] std::size_t liveCount(const TrafficVoiceAllocator& allocator)
{
    return static_cast<std::size_t>(std::count_if(allocator.voices().begin(), allocator.voices().end(),
                                                  [](const TrafficVoice& voice) { return voice.live; }));
}

} // namespace

TEST_CASE("the nearest cars inside the acquire radius take the voices and fade in", "[audio][traffic-audio]")
{
    auto allocator = TrafficVoiceAllocator(twoVoices(), {});
    const auto listener = AudioListener{};

    const auto cars = std::vector<TrafficAudioCar>{carAt(1, 200.0), carAt(2, 10.0), carAt(3, 120.0), carAt(4, 50.0)};

    REQUIRE(allocator.voices().size() == 2);

    allocator.update(cars, listener, 0.05);

    // Two voices, the two nearest cars, both started silent.
    CHECK(liveCount(allocator) == 2);
    REQUIRE(voiceFor(allocator, 2) != nullptr);
    REQUIRE(voiceFor(allocator, 4) != nullptr);
    CHECK(voiceFor(allocator, 1) == nullptr);
    CHECK(voiceFor(allocator, 3) == nullptr);
    CHECK(voiceFor(allocator, 2)->gain == Approx(0.0));

    // Half up after one fade step, full after two, and then held there.
    allocator.update(cars, listener, 0.05);
    CHECK(voiceFor(allocator, 2)->gain == Approx(0.5));
    allocator.update(cars, listener, 0.05);
    CHECK(voiceFor(allocator, 2)->gain == Approx(1.0));
    allocator.update(cars, listener, 0.05);
    CHECK(voiceFor(allocator, 2)->gain == Approx(1.0));

    // The voice carries the car's pose for the backend to place.
    CHECK(voiceFor(allocator, 4)->positionMetres.z == Approx(50.0));
    CHECK(voiceFor(allocator, 4)->body == 0);
}

TEST_CASE("a voiced car keeps its voice out to the release radius and then fades out", "[audio][traffic-audio]")
{
    auto allocator = TrafficVoiceAllocator(twoVoices(), {});
    const auto listener = AudioListener{};

    auto cars = std::vector<TrafficAudioCar>{carAt(1, 10.0), carAt(2, 50.0), carAt(3, 120.0)};

    for (auto step = 0; step < 3; step++)
    {
        allocator.update(cars, listener, 0.05);
    }

    REQUIRE(voiceFor(allocator, 2) != nullptr);
    CHECK(voiceFor(allocator, 2)->gain == Approx(1.0));

    // Between the two radii: still voiced, still full.
    cars[1] = carAt(2, 130.0);
    allocator.update(cars, listener, 0.05);
    REQUIRE(voiceFor(allocator, 2) != nullptr);
    CHECK(voiceFor(allocator, 2)->gain == Approx(1.0));

    // Past the release radius: fading, on a source that keeps moving, then gone.
    cars[1] = carAt(2, 150.0);
    allocator.update(cars, listener, 0.05);
    REQUIRE(voiceFor(allocator, 2) != nullptr);
    CHECK(voiceFor(allocator, 2)->gain == Approx(0.5));
    CHECK(voiceFor(allocator, 2)->positionMetres.z == Approx(150.0));

    allocator.update(cars, listener, 0.05);
    CHECK(voiceFor(allocator, 2) == nullptr);
    CHECK(liveCount(allocator) == 1);

    // The car at 120 is inside the release radius and outside the acquire one, so the free voice
    // stays free until something comes within a hundred metres.
    allocator.update(cars, listener, 0.05);
    CHECK(voiceFor(allocator, 3) == nullptr);

    cars[2] = carAt(3, 90.0);
    allocator.update(cars, listener, 0.05);
    CHECK(voiceFor(allocator, 3) != nullptr);
}

TEST_CASE("a much nearer car takes the farthest voice, one handover at a time", "[audio][traffic-audio]")
{
    auto allocator = TrafficVoiceAllocator(twoVoices(), {});
    const auto listener = AudioListener{};

    auto cars = std::vector<TrafficAudioCar>{carAt(1, 60.0), carAt(2, 80.0)};

    for (auto step = 0; step < 3; step++)
    {
        allocator.update(cars, listener, 0.05);
    }

    REQUIRE(voiceFor(allocator, 1) != nullptr);
    REQUIRE(voiceFor(allocator, 2) != nullptr);

    // Twenty metres is under 0.7 of eighty: the car at eighty is asked to go. The one at sixty is
    // not — a release in progress is the pending handover.
    cars.push_back(carAt(3, 20.0));
    allocator.update(cars, listener, 0.05);
    CHECK(voiceFor(allocator, 1)->gain == Approx(1.0));
    CHECK(voiceFor(allocator, 2)->gain == Approx(1.0));

    allocator.update(cars, listener, 0.05);
    CHECK(voiceFor(allocator, 1)->gain == Approx(1.0));
    REQUIRE(voiceFor(allocator, 2) != nullptr);
    CHECK(voiceFor(allocator, 2)->gain == Approx(0.5));

    allocator.update(cars, listener, 0.05);
    CHECK(voiceFor(allocator, 2) == nullptr);
    REQUIRE(voiceFor(allocator, 3) != nullptr);
    CHECK(voiceFor(allocator, 3)->gain == Approx(0.0));

    // Through the whole handover the car at sixty was never touched.
    CHECK(voiceFor(allocator, 1)->gain == Approx(1.0));

    // And a car that is nearer but not by the ratio takes nothing from anyone.
    for (auto step = 0; step < 3; step++)
    {
        allocator.update(cars, listener, 0.05);
    }

    cars.push_back(carAt(4, 50.0));
    for (auto step = 0; step < 4; step++)
    {
        allocator.update(cars, listener, 0.05);
    }

    CHECK(voiceFor(allocator, 4) == nullptr);
    CHECK(voiceFor(allocator, 1) != nullptr);
    CHECK(voiceFor(allocator, 3) != nullptr);
}

TEST_CASE("a car that vanishes releases its voice, and a slot handed on starts its gearbox afresh",
          "[audio][traffic-audio]")
{
    auto allocator = TrafficVoiceAllocator(twoVoices(), {TrafficEngineRange{.idleRpm = 700.0, .limiterRpm = 6000.0}});
    const auto listener = AudioListener{};

    // One fast car, voiced and in a high gear.
    auto cars = std::vector<TrafficAudioCar>{carAt(1, 30.0, 0, 40.0)};

    for (auto step = 0; step < 3; step++)
    {
        allocator.update(cars, listener, 0.05);
    }

    REQUIRE(voiceFor(allocator, 1) != nullptr);
    CHECK(voiceFor(allocator, 1)->state.gear > 3);
    CHECK(voiceFor(allocator, 1)->state.idleRpm == Approx(700.0));
    CHECK(voiceFor(allocator, 1)->velocityMetresPerSecond.z == Approx(40.0));

    // It ceases to exist: the voice fades where it last was and then frees.
    cars.clear();
    allocator.update(cars, listener, 0.05);
    REQUIRE(voiceFor(allocator, 1) != nullptr);
    CHECK(voiceFor(allocator, 1)->gain == Approx(0.5));
    allocator.update(cars, listener, 0.05);
    CHECK(liveCount(allocator) == 0);

    // A stationary car takes the slot and is in first at idle, not in the last car's gear.
    cars.push_back(carAt(2, 20.0, 0, 0.0));
    allocator.update(cars, listener, 0.05);
    REQUIRE(voiceFor(allocator, 2) != nullptr);
    CHECK(voiceFor(allocator, 2)->state.gear == 1);
    CHECK(voiceFor(allocator, 2)->state.engineRpm == Approx(700.0));

    // A body past the end of the stated ranges takes the default range rather than reading off it.
    cars.push_back(carAt(3, 25.0, 7, 0.0));
    allocator.update(cars, listener, 0.05);
    REQUIRE(voiceFor(allocator, 3) != nullptr);
    CHECK(voiceFor(allocator, 3)->body == 7);
    CHECK(voiceFor(allocator, 3)->state.idleRpm == Approx(TrafficEngineRange{}.idleRpm));
}
