// TrafficAudio bodies. Declarations are in Audio/Api/TrafficAudio.cppm.
//
// A **module implementation unit** — `module raceengine.audio;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

module raceengine.audio;

import :TrafficAudio;
import :CarAudio;
import :EngineLayers;
import :TyreLayers;

namespace raceengine
{

namespace
{

// A name split on the two separators the fleet's banks use: the underscore every convention uses,
// and the space the rpm-named sets put between their ordinal and the rest.
[[nodiscard]] std::vector<std::string_view> tokensOf(const std::string_view name)
{
    auto tokens = std::vector<std::string_view>{};
    auto start = std::size_t{0};

    for (auto index = std::size_t{0}; index <= name.size(); index++)
    {
        if (index == name.size() || name[index] == '_' || name[index] == ' ')
        {
            if (index > start)
            {
                tokens.push_back(name.substr(start, index - start));
            }

            start = index + 1;
        }
    }

    return tokens;
}

[[nodiscard]] bool allDigits(const std::string_view token)
{
    return !token.empty() && std::all_of(token.begin(), token.end(),
                                         [](const char character)
                                         { return std::isdigit(static_cast<unsigned char>(character)) != 0; });
}

// A token with its take number taken off: `low3` is `low`, `idle2` is `idle`, `onverylow4` is
// `onverylow`. A token that is all digits is left whole, because that is an rpm and not a take.
[[nodiscard]] std::string_view withoutTake(const std::string_view token)
{
    if (allDigits(token))
    {
        return token;
    }

    auto end = token.size();
    while (end > 0 && std::isdigit(static_cast<unsigned char>(token[end - 1])) != 0)
    {
        end--;
    }

    return token.substr(0, end);
}

[[nodiscard]] std::size_t bandOf(const std::string_view word)
{
    for (auto candidate = std::size_t{0}; candidate < engineBands.size(); candidate++)
    {
        if (word == engineBands[candidate])
        {
            return candidate;
        }
    }

    return engineBands.size();
}

// `EngA`, `EngB`, `ExhL`, `Exh0` and their bare forms, lowercased: the family a recording belongs
// to in the rpm-named banks. `engine` is not one — the family words are short by construction.
[[nodiscard]] bool isFamily(const std::string_view token)
{
    return token.size() >= 3 && token.size() <= 4 && (token.starts_with("eng") || token.starts_with("exh"));
}

struct Candidate
{
    EngineLayer layer;
    std::string family;
};

[[nodiscard]] double parseRpm(const std::string_view digits)
{
    auto value = 0.0;
    for (const auto character : digits)
    {
        value = value * 10.0 + static_cast<double>(character - '0');
    }

    return value;
}

} // namespace

[[nodiscard]] std::vector<EngineLayer> classifyTrafficEngineLayers(const std::span<const std::string> names,
                                                                   const double idleRpm, const double limiterRpm)
{
    auto candidates = std::vector<Candidate>{};

    for (auto index = std::size_t{0}; index < names.size(); index++)
    {
        auto lowered = names[index];
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](const unsigned char character) { return static_cast<char>(std::tolower(character)); });

        auto layer = EngineLayer{};
        auto family = std::string{};
        auto sawPosition = false;
        auto sawThrottle = false;
        auto sawBand = false;
        auto numericBand = false;
        auto band = std::size_t{0};
        auto numericRpm = 0.0;

        for (const auto token : tokensOf(lowered))
        {
            if (token == "ex" || token == "ext" || token == "exterior")
            {
                if (!sawPosition)
                {
                    layer.position = LayerPosition::Exterior;
                    sawPosition = true;
                }

                continue;
            }

            if (token == "in" || token == "int" || token == "interior")
            {
                if (!sawPosition)
                {
                    layer.position = LayerPosition::Interior;
                    sawPosition = true;
                }

                continue;
            }

            if (token == "on")
            {
                layer.throttle = LayerThrottle::On;
                sawThrottle = true;

                continue;
            }

            if (token == "off")
            {
                layer.throttle = LayerThrottle::Off;
                sawThrottle = true;

                continue;
            }

            if (isFamily(token))
            {
                family = std::string(token);

                if (!sawPosition)
                {
                    layer.position = token.starts_with("exh") ? LayerPosition::Exterior : LayerPosition::Interior;
                    sawPosition = true;
                }

                continue;
            }

            // The rpm the recording was made at, in the sets that say. Three digits or more, so the
            // ordinal in `5 EngA_04364` and the `0 5 0` of a turbo sample's own suffix are not read
            // as speeds. Kept whether it comes before or after the state word and judged at the end.
            if (allDigits(token))
            {
                if (token.size() >= 3)
                {
                    numericRpm = parseRpm(token);
                }

                continue;
            }

            // `ext500ss`: the outside marked as a prefix on the engine's own name.
            if (token.starts_with("ext") && token.size() > 3)
            {
                if (!sawPosition)
                {
                    layer.position = LayerPosition::Exterior;
                    sawPosition = true;
                }

                continue;
            }

            const auto stem = withoutTake(token);

            if (const auto named = bandOf(stem); named < engineBands.size())
            {
                band = named;
                sawBand = true;

                continue;
            }

            // The fused forms, `onhigh` and `offverylow`, with the take already stripped.
            if (stem.starts_with("on"))
            {
                if (const auto named = bandOf(stem.substr(2)); named < engineBands.size())
                {
                    layer.throttle = LayerThrottle::On;
                    sawThrottle = true;
                    band = named;
                    sawBand = true;
                }

                continue;
            }

            if (stem.starts_with("off"))
            {
                if (const auto named = bandOf(stem.substr(3)); named < engineBands.size())
                {
                    layer.throttle = LayerThrottle::Off;
                    sawThrottle = true;
                    band = named;
                    sawBand = true;
                }

                continue;
            }
        }

        // A bare number is a speed only when something else says the recording is an engine's: a
        // family word or a throttle word. `7025` on its own is a sample this engine has no use for.
        if (!sawBand && numericRpm > 0.0 && (!family.empty() || sawThrottle))
        {
            sawBand = true;
            numericBand = true;
        }

        if (!sawBand)
        {
            continue;
        }

        // The Golf classifier's rule, kept: `idle` with no throttle word is the engine sitting there,
        // and any other named band needs a throttle word to mean anything.
        if (!numericBand)
        {
            if (band == 0 && !sawThrottle)
            {
                layer.throttle = LayerThrottle::Idle;
            }
            else if (!sawThrottle)
            {
                continue;
            }
        }

        // No position word is the inside: every convention here names the outside and leaves the
        // inside unmarked.
        if (!sawPosition)
        {
            layer.position = LayerPosition::Interior;
        }

        layer.name = names[index];
        layer.sample = static_cast<int>(index);
        layer.centreRpm = numericBand ? numericRpm : bandCentreRpm(band, idleRpm, limiterRpm);

        candidates.push_back(Candidate{.layer = std::move(layer), .family = std::move(family)});
    }

    // One family per bank. The rpm-named banks carry the same instant recorded two or three ways —
    // the engine bay and the tailpipe, or two microphones on the engine — and crossfading between
    // recordings of the same speed is not a mix. The largest exterior family is what a car on the
    // street sounds like; the rest are dropped here rather than opened and left silent.
    {
        auto families = std::vector<std::pair<std::string, std::size_t>>{};
        auto exteriorFamilies = std::vector<std::string>{};

        for (const auto& candidate : candidates)
        {
            if (candidate.family.empty())
            {
                continue;
            }

            auto found = std::find_if(families.begin(), families.end(),
                                      [&candidate](const auto& entry) { return entry.first == candidate.family; });

            if (found == families.end())
            {
                families.emplace_back(candidate.family, std::size_t{1});
            }
            else
            {
                found->second++;
            }

            if (candidate.layer.position == LayerPosition::Exterior &&
                std::find(exteriorFamilies.begin(), exteriorFamilies.end(), candidate.family) == exteriorFamilies.end())
            {
                exteriorFamilies.push_back(candidate.family);
            }
        }

        if (!families.empty())
        {
            auto chosen = std::string{};
            auto chosenCount = std::size_t{0};

            for (const auto& [family, count] : families)
            {
                const auto exterior =
                    std::find(exteriorFamilies.begin(), exteriorFamilies.end(), family) != exteriorFamilies.end();
                const auto chosenExterior =
                    std::find(exteriorFamilies.begin(), exteriorFamilies.end(), chosen) != exteriorFamilies.end();

                if (chosen.empty() || (exterior && !chosenExterior) ||
                    (exterior == chosenExterior && count > chosenCount))
                {
                    chosen = family;
                    chosenCount = count;
                }
            }

            std::erase_if(candidates,
                          [&chosen](const Candidate& candidate)
                          { return !candidate.family.empty() && candidate.family != chosen; });
        }
    }

    auto layers = std::vector<EngineLayer>{};
    layers.reserve(candidates.size());

    for (auto& candidate : candidates)
    {
        // Alternate takes dropped, for the reason the Golf's are: a take that can never be selected
        // is a decoded sample and a channel doing nothing.
        const auto already = std::find_if(layers.begin(), layers.end(),
                                          [&candidate](const EngineLayer& seen)
                                          {
                                              return seen.position == candidate.layer.position &&
                                                     seen.throttle == candidate.layer.throttle &&
                                                     seen.centreRpm == candidate.layer.centreRpm;
                                          });

        if (already != layers.end())
        {
            continue;
        }

        layers.push_back(std::move(candidate.layer));
    }

    // A bank recorded only from the driver's seat is still heard from the pavement.
    const auto anyExterior = std::any_of(layers.begin(), layers.end(), [](const EngineLayer& layer)
                                         { return layer.position == LayerPosition::Exterior; });

    if (!anyExterior)
    {
        for (auto& layer : layers)
        {
            layer.position = LayerPosition::Exterior;
        }
    }

    return layers;
}

[[nodiscard]] std::vector<TyreLayer> classifyTrafficTyreLayers(const std::span<const std::string> names)
{
    auto layers = classifyTyreLayers(names);

    std::erase_if(layers, [](const TyreLayer& layer) { return layer.noise != TyreNoise::Rolling; });

    return layers;
}

void mixTrafficEngineLayers(const std::span<const EngineLayer> layers, const CarAudioState& car, const double idleRpm,
                            const std::span<LayerMix> out)
{
    // Judged over the exterior layers alone, because that is the side the mix is of. The Charger's
    // bank carries three interior overrun loops (`ls9_offhigh`, `ls9_offlow`, `ls9_offmid`) beside an
    // exterior set recorded on the throttle only: counted, they would hold the load at the driver's
    // fraction and hand the rest of the weight to an overrun set the street cannot hear, and the car
    // would go quiet at cruise. The Legacy's idle loops are interior for the same reason.
    auto hasIdle = false;
    auto hasOn = false;
    auto hasOff = false;

    for (const auto& layer : layers)
    {
        if (layer.position != LayerPosition::Exterior)
        {
            continue;
        }

        hasIdle = hasIdle || layer.throttle == LayerThrottle::Idle;
        hasOn = hasOn || layer.throttle == LayerThrottle::On;
        hasOff = hasOff || layer.throttle == LayerThrottle::Off;
    }

    // With no overrun set the on-throttle set is the whole engine, so it carries the whole of the
    // load rather than the fraction the driver is asking for; the mirror case for a bank of
    // overrun recordings alone.
    auto adjusted = car;
    if (!hasOff)
    {
        adjusted.load = 1.0;
    }
    else if (!hasOn)
    {
        adjusted.load = 0.0;
    }

    // With no idle recording the idle state is handed nothing: an idle speed of zero puts the
    // engine "past idle" at any speed, so the lowest recording carries the idle, pitched down to it.
    mixEngineLayers(layers, adjusted, LayerPosition::Exterior, hasIdle ? idleRpm : 0.0, out);
}

[[nodiscard]] CarAudioState deriveTrafficAudio(const TrafficGearbox& gearbox, TrafficDriveState& drive,
                                               const double speedMetresPerSecond,
                                               const double accelerationMetresPerSecondSquared, const double idleRpm,
                                               const double limiterRpm)
{
    const auto speed = std::isfinite(speedMetresPerSecond) ? std::max(0.0, speedMetresPerSecond) : 0.0;
    const auto acceleration =
        std::isfinite(accelerationMetresPerSecondSquared) ? accelerationMetresPerSecondSquared : 0.0;
    const auto idle = std::max(idleRpm, 1.0);
    const auto limiter = std::max(limiterRpm, idle * 2.0);

    const auto eager = std::clamp(acceleration / 2.5, 0.0, 1.0);
    const auto shiftUp = limiter * (gearbox.shiftUpFraction + gearbox.eagerShiftFraction * eager);
    const auto shiftDown = limiter * gearbox.shiftDownFraction;

    const auto last = gearbox.rpmPerMetrePerSecond.size() - 1;
    drive.gear = std::min(drive.gear, last);

    // Below the speed a gear idles at the clutch is slipping and the engine sits at idle, which is
    // also what a car creeping in a queue sounds like.
    const auto rpmIn = [&](const std::size_t gear)
    {
        return std::max(idle, speed * gearbox.rpmPerMetrePerSecond[gear]);
    };

    while (drive.gear < last && rpmIn(drive.gear) > shiftUp)
    {
        drive.gear++;
    }

    while (drive.gear > 0 && rpmIn(drive.gear) < shiftDown)
    {
        drive.gear--;
    }

    auto state = CarAudioState{};
    state.engineRpm = std::min(rpmIn(drive.gear), limiter);
    state.idleRpm = idle;
    state.limiterRpm = limiter;
    state.gear = static_cast<std::int32_t>(drive.gear + 1);

    // A car holding its speed is under light load — it is pushing air and rolling resistance and
    // nothing else — and the on/off blend reads the driver's acceleration around that.
    const auto load = std::clamp(0.35 + acceleration / std::max(gearbox.fullLoadAcceleration, 1e-3), 0.0, 1.0);
    state.throttle = load;
    state.load = load;

    state.roadSpeed = speed;
    state.rollingSpeed = speed;
    state.wheelSlip = 0.0;
    state.slipLoad = 0.0;

    return state;
}

TrafficVoiceAllocator::TrafficVoiceAllocator(TrafficAudioSettings settings, std::vector<TrafficEngineRange> bodies) :
    options(std::move(settings)),
    ranges(std::move(bodies)),
    slots(options.voices),
    published(options.voices)
{
}

const TrafficEngineRange& TrafficVoiceAllocator::rangeFor(const std::uint8_t body) const
{
    static constexpr auto fallback = TrafficEngineRange{};

    return body < ranges.size() ? ranges[body] : fallback;
}

void TrafficVoiceAllocator::assign(Slot& slot, const TrafficAudioCar& car, const double distanceMetres)
{
    slot.voice = TrafficVoice{};
    slot.voice.live = true;
    slot.voice.car = car.id;
    slot.voice.body = car.body;
    slot.voice.gain = 0.0;
    slot.drive = TrafficDriveState{};
    slot.releasing = false;

    refresh(slot, car, distanceMetres);
}

void TrafficVoiceAllocator::refresh(Slot& slot, const TrafficAudioCar& car, const double distanceMetres)
{
    const auto& range = rangeFor(car.body);

    slot.voice.positionMetres = car.positionMetres;
    slot.voice.velocityMetresPerSecond = car.velocityMetresPerSecond;
    slot.voice.siren = car.siren;
    slot.distanceMetres = distanceMetres;
    slot.voice.state =
        deriveTrafficAudio(options.gearbox, slot.drive, glm::length(car.velocityMetresPerSecond),
                           car.accelerationMetresPerSecondSquared, range.idleRpm, range.limiterRpm);
}

void TrafficVoiceAllocator::update(const std::span<const TrafficAudioCar> cars, const AudioListener& listener,
                                   const double deltaSeconds)
{
    const auto fadeStep = options.fadeSeconds > 0.0 ? std::max(0.0, deltaSeconds) / options.fadeSeconds : 1.0;

    voiced.assign(cars.size(), false);

    // Keep or release. A voiced car keeps its voice while it is inside the release radius and still
    // exists; a voice on its way out keeps following its car so the fade is on a moving source.
    for (auto& slot : slots)
    {
        if (!slot.voice.live)
        {
            continue;
        }

        auto found = cars.size();
        for (auto index = std::size_t{0}; index < cars.size(); index++)
        {
            if (cars[index].id == slot.voice.car)
            {
                found = index;
                break;
            }
        }

        // Weighed: a siren reads as nearer than it is, here and in every comparison below.
        const auto distance = found < cars.size()
                                  ? glm::length(cars[found].positionMetres - listener.positionMetres) *
                                        (cars[found].siren ? options.sirenDistanceWeight : 1.0)
                                  : std::numeric_limits<double>::infinity();

        if (found == cars.size() || distance > options.releaseMetres)
        {
            slot.releasing = true;
        }

        if (found < cars.size())
        {
            voiced[found] = true;
            refresh(slot, cars[found], distance);
        }

        if (!slot.releasing)
        {
            slot.voice.gain = std::min(1.0, slot.voice.gain + fadeStep);

            continue;
        }

        slot.voice.gain -= fadeStep;
        if (slot.voice.gain <= 0.0)
        {
            slot.voice = TrafficVoice{};
            slot.releasing = false;
            slot.distanceMetres = 0.0;
        }
    }

    // The unvoiced cars within reach, nearest first.
    candidates.clear();

    for (auto index = std::size_t{0}; index < cars.size(); index++)
    {
        if (voiced[index])
        {
            continue;
        }

        const auto distance = glm::length(cars[index].positionMetres - listener.positionMetres) *
                              (cars[index].siren ? options.sirenDistanceWeight : 1.0);
        if (distance <= options.acquireMetres)
        {
            candidates.emplace_back(index, distance);
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& left, const auto& right) { return left.second < right.second; });

    auto next = std::size_t{0};

    for (auto& slot : slots)
    {
        if (slot.voice.live || next >= candidates.size())
        {
            continue;
        }

        assign(slot, cars[candidates[next].first], candidates[next].second);
        next++;
    }

    // A car left over that is much nearer than the farthest voiced car takes that car's voice —
    // released now, taken on the tick the fade finishes. **One handover at a time**: while any slot
    // is fading out, the same near car is still a leftover on every tick of that fade, and re-running
    // the test would release the next-farthest voice each tick until the fade had emptied every slot
    // for one arrival. A release in progress is the pending handover, so nothing else is asked to go.
    const auto anyReleasing = std::any_of(slots.begin(), slots.end(),
                                          [](const Slot& slot) { return slot.voice.live && slot.releasing; });

    if (next < candidates.size() && !anyReleasing)
    {
        auto farthest = slots.size();

        for (auto index = std::size_t{0}; index < slots.size(); index++)
        {
            const auto& slot = slots[index];
            if (!slot.voice.live)
            {
                continue;
            }

            if (farthest == slots.size() || slot.distanceMetres > slots[farthest].distanceMetres)
            {
                farthest = index;
            }
        }

        if (farthest < slots.size() && candidates[next].second < options.preemptRatio * slots[farthest].distanceMetres)
        {
            slots[farthest].releasing = true;
        }
    }

    for (auto index = std::size_t{0}; index < slots.size(); index++)
    {
        published[index] = slots[index].voice;
    }
}

} // namespace raceengine
