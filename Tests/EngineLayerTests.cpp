#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

import raceengine;

using raceengine::bandCentreRpm;
using raceengine::CarAudioState;
using raceengine::classifyEngineLayers;
using raceengine::engineBands;
using raceengine::EngineLayer;
using raceengine::LayerMix;
using raceengine::LayerPosition;
using raceengine::LayerThrottle;
using raceengine::mixEngineLayers;

namespace
{

constexpr auto idleRpm = 850.0;
constexpr auto limiterRpm = 6800.0;

// The engine layers out of the Golf's own bank, with the samples that are not engine layers left in
// so the classifier has to reject them.
[[nodiscard]] std::vector<std::string> golfSamples()
{
    return {"4c_ex_on_idle",
            "4c_ex_on_verylow",
            "4c_ex_on_low_pr",
            "4c_ex_on_mid_pr2",
            "4c_ex_on_high",
            "4c_ex_off_verylow",
            "4c_ex_off_low",
            "4c_ex_off_mid",
            "4c_ex_off_high",
            "4c_ex_idle",
            "4c_in_on_verylow",
            "4c_in_on_low_2",
            "4c_in_on_mid_pr",
            "4c_in_on_high",
            "4c_in_off_low",
            "4c_in_idle",
            "horn",
            "door_open",
            "backfire_5",
            "wind",
            "tyre_rolling",
            "Skid",
            "500_limiter",
            "turbo"};
}

[[nodiscard]] CarAudioState driving(const double rpm, const double load)
{
    auto car = CarAudioState{};
    car.engineRpm = rpm;
    car.load = load;
    car.throttle = load;

    return car;
}

[[nodiscard]] double totalGain(const std::vector<LayerMix>& mix)
{
    return std::accumulate(mix.begin(), mix.end(), 0.0,
                           [](const double sum, const LayerMix& one) { return sum + one.gain; });
}

} // namespace

TEST_CASE("only engine layers are classified as engine layers", "[audio][layers]")
{
    const auto names = golfSamples();
    const auto layers = classifyEngineLayers(names, idleRpm, limiterRpm);

    SECTION("the engine ones are found")
    {
        REQUIRE(layers.size() == 16);
    }

    SECTION("and nothing else is")
    {
        // A door slam in the engine mix is a car that thumps once per crossfade. The classifier takes
        // only names that carry a position, a state and a band, so everything else is simply absent
        // rather than guessed at.
        for (const auto& layer : layers)
        {
            REQUIRE(layer.name.starts_with("4c_"));
        }
    }

    SECTION("position and state come off the name")
    {
        const auto find = [&layers](const std::string& name)
        {
            return std::find_if(layers.begin(), layers.end(), [&name](const EngineLayer& l) { return l.name == name; });
        };

        REQUIRE(find("4c_ex_on_high")->position == LayerPosition::Exterior);
        REQUIRE(find("4c_ex_on_high")->throttle == LayerThrottle::On);
        REQUIRE(find("4c_in_off_low")->position == LayerPosition::Interior);
        REQUIRE(find("4c_in_off_low")->throttle == LayerThrottle::Off);

        // `idle` is the one word that is both a band and a state, and which it means is decided by
        // what else the name carries. Read as the state unconditionally, the on-throttle recording at
        // the bottom of the range collides with the idle loop at one band and whichever the bank
        // happens to list first is what the car idles to.
        REQUIRE(find("4c_ex_idle")->throttle == LayerThrottle::Idle);
        REQUIRE(find("4c_ex_on_idle")->throttle == LayerThrottle::On);
        REQUIRE(find("4c_ex_on_idle")->centreRpm == find("4c_ex_idle")->centreRpm);

        // A suffix the author added does not stop it being a layer: `_pr` and `_2` are the same
        // recording, longer or alternate.
        REQUIRE(find("4c_ex_on_low_pr") != layers.end());
        REQUIRE(find("4c_in_on_low_2") != layers.end());
    }
}

TEST_CASE("an alternate take is dropped rather than opened and left silent", "[audio][layers]")
{
    // The Golf's own bank carries `4c_ex_idle` and `4c_ex_idle_2`, the same length to the byte. They
    // are meant to be picked between at random, and picking at random means restarting a loop — which
    // this mixer never does. So the second can never be selected, and keeping it would be a decoded
    // sample, a channel doing nothing, and a load log naming a layer nobody will hear.
    const auto names = std::vector<std::string>{"4c_ex_idle", "4c_ex_idle_2", "4c_in_on_high", "4c_in_on_high_4"};

    const auto layers = classifyEngineLayers(names, idleRpm, limiterRpm);

    REQUIRE(layers.size() == 2);
    REQUIRE(layers[0].name == "4c_ex_idle");
    REQUIRE(layers[1].name == "4c_in_on_high");

    SECTION("but a take that means something different is not a duplicate")
    {
        // Same band, same side of the glass, different throttle — three layers, not one.
        const auto distinct = std::vector<std::string>{"4c_ex_on_high", "4c_ex_off_high", "4c_in_on_high"};

        REQUIRE(classifyEngineLayers(distinct, idleRpm, limiterRpm).size() == 3);
    }
}

TEST_CASE("the bands are spread in octaves rather than evenly", "[audio][layers]")
{
    // An engine's note is heard in octaves: 1000 to 2000 rpm is the same musical distance as 3000 to
    // 6000. Spread linearly, every recording bunches into the top of the range and one loop is
    // stretched across the bottom octave and a half — which is what the cheap version sounds like.
    auto centres = std::vector<double>{};
    for (auto band = std::size_t{0}; band < engineBands.size(); band++)
    {
        centres.push_back(bandCentreRpm(band, idleRpm, limiterRpm));
    }

    REQUIRE(centres.front() == idleRpm);
    REQUIRE(centres.back() == Catch::Approx(limiterRpm));

    SECTION("each step up is the same ratio, not the same number of rpm")
    {
        const auto firstRatio = centres[2] / centres[1];
        const auto lastRatio = centres[5] / centres[4];

        REQUIRE(firstRatio == Catch::Approx(lastRatio).epsilon(1e-9));

        // And that means the gaps in rpm grow, which is the whole point.
        REQUIRE(centres[5] - centres[4] > centres[2] - centres[1]);
    }
}

TEST_CASE("the engine crossfades between layers rather than switching", "[audio][layers]")
{
    const auto names = golfSamples();
    const auto layers = classifyEngineLayers(names, idleRpm, limiterRpm);
    auto mix = std::vector<LayerMix>(layers.size());

    SECTION("two layers sound at once between their centres")
    {
        // Halfway between two recordings both are heard. One alone would be a switch, and a switch is
        // audible as a step in timbre at exactly the same speed every time.
        const auto low = bandCentreRpm(2, idleRpm, limiterRpm);
        const auto mid = bandCentreRpm(3, idleRpm, limiterRpm);

        mixEngineLayers(layers, driving(std::sqrt(low * mid), 1.0), LayerPosition::Exterior, idleRpm, mix);

        auto sounding = 0;
        for (const auto& one : mix)
        {
            sounding += one.gain > 0.01 ? 1 : 0;
        }

        REQUIRE(sounding >= 2);
    }

    SECTION("and the crossfade holds its power rather than dipping through the middle")
    {
        // Two loops of the same engine are uncorrelated, so their powers add. A linear pair loses
        // three decibels in the middle of every crossfade, which is a soft spot at exactly the speeds
        // a driver spends most of their time at.
        const auto low = bandCentreRpm(2, idleRpm, limiterRpm);
        const auto mid = bandCentreRpm(3, idleRpm, limiterRpm);

        auto worst = 1.0;
        for (auto step = 0; step <= 20; step++)
        {
            const auto through = static_cast<double>(step) / 20.0;
            const auto rpm = low * std::pow(mid / low, through);

            mixEngineLayers(layers, driving(rpm, 1.0), LayerPosition::Exterior, idleRpm, mix);

            auto power = 0.0;
            for (const auto& one : mix)
            {
                power += one.gain * one.gain;
            }

            worst = std::min(worst, power);
        }

        // Equal power keeps the sum of squares at one across the whole sweep; linear would drop to
        // one half.
        REQUIRE(worst > 0.9);
    }
}

TEST_CASE("pitch carries the note between recordings", "[audio][layers]")
{
    // Crossfade alone steps in *pitch*: the note would sit still at each recording's own speed and
    // jump between them. What makes it rise smoothly is each layer being resampled toward where the
    // engine actually is.
    const auto names = golfSamples();
    const auto layers = classifyEngineLayers(names, idleRpm, limiterRpm);
    auto mix = std::vector<LayerMix>(layers.size());

    const auto loudestPitch = [&](const double rpm)
    {
        mixEngineLayers(layers, driving(rpm, 1.0), LayerPosition::Exterior, idleRpm, mix);

        auto best = 0.0;
        auto pitch = 0.0;
        for (const auto& one : mix)
        {
            if (one.gain > best)
            {
                best = one.gain;
                pitch = one.pitch;
            }
        }

        return pitch;
    };

    const auto atCentre = bandCentreRpm(3, idleRpm, limiterRpm);

    // At a recording's own speed it plays as recorded.
    REQUIRE(loudestPitch(atCentre) == Catch::Approx(1.0).epsilon(0.02));

    // Above it, faster.
    REQUIRE(loudestPitch(atCentre * 1.15) > 1.0);

    SECTION("but never so far that it stops being the engine")
    {
        // A recording stretched past an octave sounds like the tape rather than the car, so past that
        // it is better to hear a neighbouring layer slightly out of tune.
        for (const auto rpm : {100.0, 500.0, 20000.0})
        {
            mixEngineLayers(layers, driving(rpm, 1.0), LayerPosition::Exterior, idleRpm, mix);

            for (const auto& one : mix)
            {
                REQUIRE(one.pitch >= 0.5);
                REQUIRE(one.pitch <= 2.0);
            }
        }
    }
}

TEST_CASE("throttle chooses between pulling and coasting", "[audio][layers]")
{
    // The same speed on and off the throttle is two completely different noises, and speed alone
    // cannot tell them apart. This is the whole reason `load` is derived at all.
    const auto names = golfSamples();
    const auto layers = classifyEngineLayers(names, idleRpm, limiterRpm);
    auto mix = std::vector<LayerMix>(layers.size());

    const auto stateGain = [&](const double load, const LayerThrottle state)
    {
        mixEngineLayers(layers, driving(3000.0, load), LayerPosition::Exterior, idleRpm, mix);

        auto sum = 0.0;
        for (auto index = std::size_t{0}; index < layers.size(); index++)
        {
            if (layers[index].throttle == state)
            {
                sum += mix[index].gain;
            }
        }

        return sum;
    };

    REQUIRE(stateGain(1.0, LayerThrottle::On) > stateGain(0.0, LayerThrottle::On));
    REQUIRE(stateGain(0.0, LayerThrottle::Off) > stateGain(1.0, LayerThrottle::Off));

    SECTION("and an on-throttle layer is never crossfaded into an overrun one")
    {
        // Blending them because they happen to be adjacent in speed is how an engine ends up sounding
        // like it is doing both at once. At full load the overrun layers are silent.
        REQUIRE(stateGain(1.0, LayerThrottle::Off) < 0.01);
    }
}

TEST_CASE("idle is its own state and lets go as the engine leaves it", "[audio][layers]")
{
    const auto names = golfSamples();
    const auto layers = classifyEngineLayers(names, idleRpm, limiterRpm);
    auto mix = std::vector<LayerMix>(layers.size());

    const auto idleGain = [&](const double rpm)
    {
        mixEngineLayers(layers, driving(rpm, 0.0), LayerPosition::Exterior, idleRpm, mix);

        auto sum = 0.0;
        for (auto index = std::size_t{0}; index < layers.size(); index++)
        {
            if (layers[index].throttle == LayerThrottle::Idle)
            {
                sum += mix[index].gain;
            }
        }

        return sum;
    };

    REQUIRE(idleGain(idleRpm) > 0.9);
    REQUIRE(idleGain(idleRpm * 2.5) < 0.01);
    REQUIRE(idleGain(idleRpm) > idleGain(idleRpm * 1.5));
}

TEST_CASE("the listener's position silences the other side rather than removing it", "[audio][layers]")
{
    // A gain of zero on a running loop is what lets the camera move between inside and outside the
    // car without restarting anything, and a loop restarted mid-note is a click.
    const auto names = golfSamples();
    const auto layers = classifyEngineLayers(names, idleRpm, limiterRpm);
    auto mix = std::vector<LayerMix>(layers.size());

    mixEngineLayers(layers, driving(3000.0, 1.0), LayerPosition::Interior, idleRpm, mix);

    REQUIRE(mix.size() == layers.size());

    for (auto index = std::size_t{0}; index < layers.size(); index++)
    {
        // Every layer is still in the mix and still names its sample.
        REQUIRE(mix[index].sample == layers[index].sample);

        if (layers[index].position == LayerPosition::Exterior)
        {
            REQUIRE(mix[index].gain == 0.0);
        }
    }

    REQUIRE(totalGain(mix) > 0.5);
}

TEST_CASE("a bank with no engine layers at all mixes to silence rather than misbehaving", "[audio][layers]")
{
    const auto names = std::vector<std::string>{"horn", "door_open", "wind"};
    const auto layers = classifyEngineLayers(names, idleRpm, limiterRpm);

    REQUIRE(layers.empty());

    auto mix = std::vector<LayerMix>{};
    mixEngineLayers(layers, driving(3000.0, 1.0), LayerPosition::Exterior, idleRpm, mix);

    REQUIRE(mix.empty());
}
