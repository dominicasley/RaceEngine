module;

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

export module raceengine.traffic:Archetypes;

namespace raceengine
{

// What kind of driver is in a traffic car.
//
// **The enum is a name and the struct below it is the behaviour**, and that split is the whole of
// how this stays extensible. Nothing in the driver model branches on the enum: it reads
// `DriverBehaviour` fields, and a population is handed a *table* of profiles with weights. So a
// fourth kind of driver — a bus that pulls over, a delivery van that double-parks, a learner — is a
// new entry in that table and a new set of numbers, not a new case in a switch somewhere inside the
// update loop. The enum exists so the three the game ships with can be named in a log line and in a
// test.
export enum class DriverArchetype : std::uint8_t
{
    Cautious,
    Regular,
    Aggressive
};

export inline constexpr std::size_t namedArchetypeCount = 3;

// How one driver drives. Every field is read by the update; none of them is read by anything else.
//
// The longitudinal half is the **Intelligent Driver Model** (Treiber, Hennecke and Helbing, 2000)
// and the lane-change half is **MOBIL** (Kesting, Treiber and Helbing, 2007). Both are named here
// rather than reinvented because both are the standard published answer to exactly this question,
// both are two dozen lines of arithmetic, and both are parameterised by quantities a person can
// argue about — a headway in seconds, a comfortable braking rate — rather than by gains that mean
// nothing outside the code they were fitted in.
export struct DriverBehaviour
{
    // --- how fast ------------------------------------------------------------------------------

    // What this driver wants, relative to the lane's own limit, m/s. Negative is a driver who will
    // not do the limit; positive is one who treats it as a suggestion.
    double speedOffsetMetresPerSecond = 0.0;
    // How much one driver of this kind differs from the next, m/s, spread evenly either side of the
    // offset above. Without it every car of a kind runs at exactly the same speed and traffic forms
    // rigid blocks that never interact.
    double speedSpreadMetresPerSecond = 1.0;

    // --- the car following model (IDM) ---------------------------------------------------------

    // The time gap this driver keeps to the car in front, seconds. This is the tailgating dial.
    double desiredHeadwaySeconds = 1.5;
    // The gap kept at a standstill, metres, bumper to bumper.
    double minimumGapMetres = 2.5;
    // What this driver treats as brisk acceleration and comfortable braking, m/s^2. The first is the
    // take-off dial; the second is how hard they will lean on the brakes before a queue.
    double comfortableAccelerationMetresPerSecondSquared = 1.6;
    double comfortableDecelerationMetresPerSecondSquared = 2.2;
    // How sharply the approach to the desired speed flattens out. Four is the value the model is
    // usually quoted with and there is no reason here to move it.
    double accelerationExponent = 4.0;
    // The ceiling on braking when a collision is actually imminent, m/s^2. The comfortable rate is
    // what a driver chooses; this is what the car can do and what stops the model driving into the
    // back of a queue it noticed late.
    double emergencyDecelerationMetresPerSecondSquared = 8.0;

    // --- reaction ------------------------------------------------------------------------------

    // How long after a light turns green before this driver moves, seconds. The "slow off the line"
    // dial, and the reason `TrafficSignal` exists at all.
    double greenDelaySeconds = 0.8;
    // The same delay applied to a queue moving off in front of them, seconds.
    double startDelaySeconds = 0.5;

    // --- the lane change model (MOBIL) ---------------------------------------------------------

    // Whether this driver changes lanes to get on, as opposed to only when the road makes them.
    // False is the cautious driver: they will still move over when their lane is about to run out,
    // because the alternative is stopping in the road.
    bool changesLanesToGetOn = true;
    // How much this driver weighs the inconvenience caused to the car they pull in front of, 0 to 1.
    // One is entirely considerate; zero is a driver who does not look.
    double politeness = 0.3;
    // How much better the new lane has to be before it is worth moving, m/s^2.
    double changeThresholdMetresPerSecondSquared = 0.2;
    // The hardest the car being pulled in front of may be forced to brake, m/s^2. This is the safety
    // criterion and it is the one number in the model that is not a preference.
    double safeDecelerationMetresPerSecondSquared = 4.0;
    // How long after finishing a change before another may start, seconds.
    double changeCooldownSeconds = 6.0;
    // How long the change itself takes, seconds — how quickly the car crosses the line.
    double changeDurationSeconds = 2.6;
    // Extra incentive to pass a car that is slower than this driver wants to go, m/s^2. Not part of
    // published MOBIL: it is what turns "the next lane is marginally freer" into "get past this
    // one", which is the difference between traffic that shuffles and traffic that overtakes.
    double overtakeUrgeMetresPerSecondSquared = 0.0;
};

// One kind of driver and how common it is.
export struct DriverProfile
{
    std::string name;
    // Relative frequency. The weights are normalised, so they are ratios rather than fractions.
    double weight = 1.0;
    DriverBehaviour behaviour;
};

// The behaviour of one of the three named kinds.
export [[nodiscard]] DriverBehaviour behaviourFor(const DriverArchetype archetype);

export [[nodiscard]] std::string_view archetypeName(const DriverArchetype archetype);

// The three the game ships with, in the mix it ships with: mostly regular, with a cautious minority
// and an aggressive one. The weights are a starting point for the seat and not a measurement — how
// much of a city's traffic should be pushy is a question about how the game feels.
export [[nodiscard]] std::vector<DriverProfile> defaultDriverProfiles();

} // namespace raceengine
