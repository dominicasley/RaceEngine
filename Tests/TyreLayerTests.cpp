#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

import raceengine;

using raceengine::CarAudioState;
using raceengine::classifyTyreLayers;
using raceengine::LayerMix;
using raceengine::LayerPosition;
using raceengine::mixTyreLayers;
using raceengine::TyreLayer;
using raceengine::TyreNoise;
using raceengine::tyreRollingReferenceSpeed;
using raceengine::tyreSkidFullSlip;
using raceengine::tyreSkidOnsetSlip;

namespace
{

// The Golf bank's own names, with the ones that must not classify left in: the tyre words that are
// not tyre noises, and a sampling of everything else the bank carries.
[[nodiscard]] std::vector<std::string> golfSamples()
{
    return {"4c_ex_on_high", "tyre_explosion", "Skid",      "flat_tyre_mono", "tyre_rolling",
            "skid_ext_mono", "wind",           "bodywork",  "horn",           "traction_control",
            "500_limiter",   "door_open",      "backfire_5"};
}

[[nodiscard]] std::vector<LayerMix> mixFor(const std::vector<TyreLayer>& layers, const CarAudioState& car,
                                           const LayerPosition listening)
{
    auto mix = std::vector<LayerMix>(layers.size());
    mixTyreLayers(layers, car, listening, mix);

    return mix;
}

[[nodiscard]] double gainOf(const std::vector<TyreLayer>& layers, const std::vector<LayerMix>& mix,
                            const std::string& name)
{
    for (auto index = std::size_t{0}; index < layers.size(); index++)
    {
        if (layers[index].name == name)
        {
            return mix[index].gain;
        }
    }

    return -1.0;
}

} // namespace

TEST_CASE("the tyre classifier takes the two noises and nothing else", "[audio][tyres]")
{
    const auto layers = classifyTyreLayers(golfSamples());

    REQUIRE(layers.size() == 3);

    // The rolling loop is one recording for both sides of the glass, so it names no side and is
    // assigned none.
    REQUIRE(layers[1].name == "tyre_rolling");
    REQUIRE(layers[1].noise == TyreNoise::Rolling);
    REQUIRE_FALSE(layers[1].positioned);

    // `skid_ext_mono` names its side; the bare `Skid` beside it is the other one, the way the
    // engine's `ex` and `in` pair.
    REQUIRE(layers[0].name == "Skid");
    REQUIRE(layers[0].noise == TyreNoise::Skid);
    REQUIRE(layers[0].positioned);
    REQUIRE(layers[0].position == LayerPosition::Interior);

    REQUIRE(layers[2].name == "skid_ext_mono");
    REQUIRE(layers[2].position == LayerPosition::Exterior);

    // `flat_tyre_mono` and `tyre_explosion` are tyre words for noises this engine does not make, and
    // looping either would be worse than silence.
    for (const auto& layer : layers)
    {
        REQUIRE(layer.name.find("flat") == std::string::npos);
        REQUIRE(layer.name.find("explosion") == std::string::npos);
    }
}

TEST_CASE("rolling follows the tread and not the car", "[audio][tyres]")
{
    const auto layers = classifyTyreLayers(golfSamples());

    auto car = CarAudioState{};

    SECTION("a parked car is silent")
    {
        const auto mix = mixFor(layers, car, LayerPosition::Exterior);

        for (const auto& entry : mix)
        {
            REQUIRE(entry.gain == 0.0);
        }
    }

    SECTION("it rises with tread speed and is full by the reference")
    {
        car.rollingSpeed = 0.25 * tyreRollingReferenceSpeed;
        car.roadSpeed = car.rollingSpeed;
        const auto quarter = gainOf(layers, mixFor(layers, car, LayerPosition::Exterior), "tyre_rolling");

        car.rollingSpeed = tyreRollingReferenceSpeed;
        car.roadSpeed = car.rollingSpeed;
        const auto full = gainOf(layers, mixFor(layers, car, LayerPosition::Exterior), "tyre_rolling");

        // The square root: audible early, saturating rather than exploding.
        REQUIRE(quarter == Catch::Approx(0.5));
        REQUIRE(full == Catch::Approx(1.0));

        car.rollingSpeed = 3.0 * tyreRollingReferenceSpeed;
        const auto past = mixFor(layers, car, LayerPosition::Exterior);
        REQUIRE(gainOf(layers, past, "tyre_rolling") == Catch::Approx(1.0));

        // And the pitch stays a recording rather than a siren.
        for (auto index = std::size_t{0}; index < layers.size(); index++)
        {
            REQUIRE(past[index].pitch >= 0.5);
            REQUIRE(past[index].pitch <= 2.0);
        }
    }

    SECTION("a locked wheel is skid without rolling, and a burnout is the reverse case")
    {
        // Locked: sliding at road speed with the tread stopped.
        car.rollingSpeed = 0.0;
        car.roadSpeed = 30.0;
        car.wheelSlip = 1.0;
        car.slipLoad = 1.0;

        const auto locked = mixFor(layers, car, LayerPosition::Exterior);
        REQUIRE(gainOf(layers, locked, "tyre_rolling") == 0.0);
        REQUIRE(gainOf(layers, locked, "skid_ext_mono") == Catch::Approx(1.0));

        // Burnout: the car stands still and the tread does everything.
        car.rollingSpeed = 15.0;
        car.roadSpeed = 0.0;

        const auto burnout = mixFor(layers, car, LayerPosition::Exterior);
        REQUIRE(gainOf(layers, burnout, "tyre_rolling") > 0.5);
        REQUIRE(gainOf(layers, burnout, "skid_ext_mono") == Catch::Approx(1.0));
    }
}

TEST_CASE("the squeal starts past the grip peak and answers the load", "[audio][tyres]")
{
    const auto layers = classifyTyreLayers(golfSamples());

    auto car = CarAudioState{};
    car.roadSpeed = 25.0;
    car.rollingSpeed = 25.0;
    car.slipLoad = 1.0;

    // Ordinary braking effort sits below the onset — a car that squealed every stop would be a car
    // that cries wolf.
    car.wheelSlip = 0.5 * tyreSkidOnsetSlip;
    REQUIRE(gainOf(layers, mixFor(layers, car, LayerPosition::Exterior), "skid_ext_mono") == 0.0);

    car.wheelSlip = tyreSkidFullSlip;
    REQUIRE(gainOf(layers, mixFor(layers, car, LayerPosition::Exterior), "skid_ext_mono") == Catch::Approx(1.0));

    // A wheel in the air spins as fast as it likes in silence.
    car.slipLoad = 0.0;
    REQUIRE(gainOf(layers, mixFor(layers, car, LayerPosition::Exterior), "skid_ext_mono") == 0.0);

    // And parking-speed slip angles, which are numerically wild and physically almost silent, fade
    // out rather than scream through a three-point turn.
    car.slipLoad = 1.0;
    car.wheelSlip = 1.0;
    car.roadSpeed = 0.4;
    car.rollingSpeed = 0.4;
    REQUIRE(gainOf(layers, mixFor(layers, car, LayerPosition::Exterior), "skid_ext_mono") == Catch::Approx(0.2));
}

TEST_CASE("the wrong side of the glass is silent, not absent", "[audio][tyres]")
{
    const auto layers = classifyTyreLayers(golfSamples());

    auto car = CarAudioState{};
    car.roadSpeed = 30.0;
    car.rollingSpeed = 30.0;
    car.wheelSlip = 1.0;
    car.slipLoad = 1.0;

    const auto outside = mixFor(layers, car, LayerPosition::Exterior);
    const auto inside = mixFor(layers, car, LayerPosition::Interior);

    // Every layer has an entry either way — a channel with no entry is a loop somebody has to stop,
    // and stopping is the one thing the mix never does.
    REQUIRE(outside.size() == layers.size());
    REQUIRE(inside.size() == layers.size());

    REQUIRE(gainOf(layers, outside, "skid_ext_mono") > 0.0);
    REQUIRE(gainOf(layers, outside, "Skid") == 0.0);

    REQUIRE(gainOf(layers, inside, "Skid") > 0.0);
    REQUIRE(gainOf(layers, inside, "skid_ext_mono") == 0.0);

    // The unpositioned rolling loop is heard from both.
    REQUIRE(gainOf(layers, outside, "tyre_rolling") > 0.0);
    REQUIRE(gainOf(layers, inside, "tyre_rolling") > 0.0);
}
