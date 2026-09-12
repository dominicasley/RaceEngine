module;

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

export module raceengine.audio:TrafficAudio;

import :CarAudio;
import :EngineLayers;
import :TyreLayers;

namespace raceengine
{

// The sound of the cars that are not the player's, and the three things that make it different
// from the player's own.
//
// **It is placed.** The player's engine is a 2D mix because the listener is inside or beside it;
// a traffic car is somewhere on the street, and what says "a car drove past" is the level falling
// with distance, the pan crossing from one side to the other and the pitch dropping through the
// pass — the Doppler shift — none of which a 2D channel can do. So a traffic voice carries a
// position and a velocity in metres, the listener carries the same, and the backend hands both to
// its 3D path.
//
// **It has no driveline.** A traffic car is a point on a lane with a speed and an acceleration, so
// the engine speed a bank wants is *derived*: a virtual gearbox that shifts up when the note gets
// high and down when it gets low, which is all a passer-by hears of a gearbox anyway. The numbers
// are a sound designer's, moved by ear, and read by nothing downstream of a speaker.
//
// **It is rationed.** Four hundred cars are simulated and a handful are audible: a voice is a set
// of looping channels, and the allocator below hands the few there are to the nearest cars with
// enough hysteresis that a car on the boundary does not flicker in and out.
//
// Pure throughout, for the reason the engine and tyre mixers are: this is the part worth being sure
// of, and none of it needs a device.

// Where the ear is. Metres, the physics frame, world relative.
export struct AudioListener
{
    glm::dvec3 positionMetres{0.0};
    // What the listener is riding in, for the Doppler shift. A camera bolted to the car moves at
    // the car's speed; a free camera hovers.
    glm::dvec3 velocityMetresPerSecond{0.0};
    glm::dvec3 forward{0.0, 0.0, 1.0};
    glm::dvec3 up{0.0, 1.0, 0.0};
};

// One recording of a fleet car, as the backend opens it: the name it is classified by and the file
// it is in. A wav each rather than the car's bank, because a traffic car wants a dozen loops out of
// the sixty a bank carries and the exporter has already written them out one to a file.
export struct TrafficSample
{
    std::string name;
    std::string path;
};

// One body shape's sound: its recordings and the two engine speeds the classifier and the gearbox
// read. Neither speed is in a bank — a bank names its layers `low` and `high`, or by the rpm the
// microphone heard, and states nothing about the engine — so both are authored beside the model.
export struct TrafficBank
{
    std::string name;
    std::vector<TrafficSample> samples;
    double idleRpm = 800.0;
    double limiterRpm = 6500.0;
};

// A fleet bank's sample names into engine layers.
//
// A superset of `classifyEngineLayers`, because the fleet's banks were not all made the way the
// Golf's was. Beside AC's `<engine>_<ex|in>_<on|off>_<band>` it reads:
//
//  - **the rpm-named sets** — `5 EngA_04364`, `13 ExhL_07540`, `8 Exh0_04328` — where a family
//    token (`EngA`, `EngB`, `ExhL`, `Exh0`) is followed by the speed the recording was made at.
//    The exhaust families are the car heard from outside and the engine families from inside;
//    when a bank carries several, the largest exterior family is kept and the rest dropped,
//    because crossfading between two recordings of the same instant is not a mix;
//  - **fused state and band** — `GC8_ex_in_onhigh`, `GC8_ex_in_offverylow` — where the throttle
//    word and the band word are one token, with an optional take number on the end;
//  - **numeric bands** — `ext500ss_on_7500` — where the band is the rpm itself;
//  - **`ext` as a prefix** — `ext500ss_idle` — and the first position word wins, so the Legacy's
//    `ex_in_` reads as exterior;
//  - **no position word at all**, which is read as interior, because every convention here names
//    the outside explicitly and the inside by omission.
//
// A bank that classifies to no exterior layer has its interior ones promoted, because a traffic
// car recorded only from the driver's seat still has to be heard from the pavement. Alternate takes
// are dropped, as the Golf's are.
export [[nodiscard]] std::vector<EngineLayer> classifyTrafficEngineLayers(std::span<const std::string> names,
                                                                          double idleRpm, double limiterRpm);

// The tread loop alone. A traffic car never slides on purpose, so the skid recordings a bank
// carries are left closed rather than looped at zero.
export [[nodiscard]] std::vector<TyreLayer> classifyTrafficTyreLayers(std::span<const std::string> names);

// The engine mix for one traffic voice, from outside the car, with the two allowances a fleet bank
// needs that the Golf's does not. A bank with no idle recording has its lowest layer carry the idle
// rather than fading to nothing there; a bank with no overrun recordings has its on-throttle set
// carry the whole of the load rather than going quiet when the driver lifts.
export void mixTrafficEngineLayers(std::span<const EngineLayer> layers, const CarAudioState& car, double idleRpm,
                                   std::span<LayerMix> out);

// The gearbox a traffic car does not have.
//
// Each gear is an engine speed per metre per second of road speed; the driver shifts up when the
// note passes one fraction of the limiter and down when it falls below another, and the gap
// between the two fractions is wider than the gap between adjacent gears so a car holding a speed
// on the boundary does not hunt. A driver who is accelerating hard holds each gear longer.
export struct TrafficGearbox
{
    std::array<double, 5> rpmPerMetrePerSecond{380.0, 230.0, 150.0, 105.0, 80.0};
    double shiftUpFraction = 0.42;
    double shiftDownFraction = 0.22;
    // How much further up the range a driver pulling 2.5 m/s^2 takes each gear.
    double eagerShiftFraction = 0.20;
    // What the driver treats as flat out and as lifting, m/s^2, for the on/off blend.
    double fullLoadAcceleration = 2.0;
};

// What one voice remembers between ticks: the gear it is in. Reset when the voice changes car.
export struct TrafficDriveState
{
    std::size_t gear = 0;
};

// A traffic car's state in the units a bank names, from what the lane model knows about it.
export [[nodiscard]] CarAudioState deriveTrafficAudio(const TrafficGearbox& gearbox, TrafficDriveState& drive,
                                                      double speedMetresPerSecond,
                                                      double accelerationMetresPerSecondSquared, double idleRpm,
                                                      double limiterRpm);

export inline constexpr std::uint32_t noTrafficCar = 0xffffffffu;

// One placed source, this tick. What a backend is handed: a body shape to pick the bank by, a pose
// and a velocity for the 3D path, a fade the allocator runs, and the car's state for the mixers.
export struct TrafficVoice
{
    bool live = false;
    // Which car this voice is following, so a backend knows when a slot changed car and has to
    // restart its channels on the other bank.
    std::uint32_t car = noTrafficCar;
    std::uint8_t body = 0;
    glm::dvec3 positionMetres{0.0};
    glm::dvec3 velocityMetresPerSecond{0.0};
    // 0 to 1 across the voice's fade in and out, multiplied into every layer's gain.
    double gain = 0.0;
    CarAudioState state{};
    // The siren, looped and placed with the car (docs/police-pursuit-brief.md).
    bool siren = false;
};

// One car as the allocator sees it, from whatever is simulating traffic.
export struct TrafficAudioCar
{
    std::uint32_t id = noTrafficCar;
    std::uint8_t body = 0;
    glm::dvec3 positionMetres{0.0};
    glm::dvec3 velocityMetresPerSecond{0.0};
    double accelerationMetresPerSecondSquared = 0.0;
    // A patrol car with its siren on.
    bool siren = false;
};

// The two engine speeds of one body shape, for the gearbox and the mix.
export struct TrafficEngineRange
{
    double idleRpm = 800.0;
    double limiterRpm = 6500.0;
};

export struct TrafficAudioSettings
{
    // How many cars are audible at once. Each is a channel per engine layer plus the tread loop.
    std::size_t voices = 8;
    // A car inside the first may take a free voice; a car past the second gives its voice back.
    // The gap between them is the hysteresis.
    double acquireMetres = 110.0;
    double releaseMetres = 140.0;
    // A car this much nearer than the farthest voiced one takes its voice, so a car passing at
    // arm's length is never silent because eight cars a street away got there first.
    double preemptRatio = 0.7;
    // How long a voice takes to come up when it starts and to go down before it stops. A loop
    // started at full gain mid-cycle is a click.
    double fadeSeconds = 0.12;
    // A car with its siren on is heard from much further than one without, and takes a voice ahead
    // of a nearer car that has none: its distance is multiplied by this before every comparison the
    // allocator makes, so a siren 400 m away reads as a car 100 m away.
    double sirenDistanceWeight = 0.25;
    TrafficGearbox gearbox{};
};

// Which cars are audible, and what each of them sounds like.
//
// Holds one slot per voice; `update` re-reads the cars every tick, keeps a voiced car's slot while
// it is inside the release radius, fades the rest out, and hands free slots to the nearest cars
// inside the acquire radius. The result is a span the backend places.
export class TrafficVoiceAllocator
{
public:
    // `bodies` is the engine range of each body shape, indexed by `TrafficAudioCar::body`; a body
    // past the end takes the default range.
    TrafficVoiceAllocator(TrafficAudioSettings settings, std::vector<TrafficEngineRange> bodies);

    void update(std::span<const TrafficAudioCar> cars, const AudioListener& listener, double deltaSeconds);

    [[nodiscard]] std::span<const TrafficVoice> voices() const
    {
        return published;
    }

    [[nodiscard]] const TrafficAudioSettings& settings() const
    {
        return options;
    }

private:
    struct Slot
    {
        TrafficVoice voice{};
        TrafficDriveState drive{};
        bool releasing = false;
        double distanceMetres = 0.0;
    };

    [[nodiscard]] const TrafficEngineRange& rangeFor(std::uint8_t body) const;
    void assign(Slot& slot, const TrafficAudioCar& car, double distanceMetres);
    void refresh(Slot& slot, const TrafficAudioCar& car, double distanceMetres);

    TrafficAudioSettings options;
    std::vector<TrafficEngineRange> ranges;
    std::vector<Slot> slots;
    std::vector<TrafficVoice> published;
    // Scratch the tick reuses, so a city full of cars allocates nothing per tick.
    std::vector<bool> voiced;
    // Unvoiced cars inside the acquire radius, as (index, distance), nearest first.
    std::vector<std::pair<std::size_t, double>> candidates;
};

} // namespace raceengine
