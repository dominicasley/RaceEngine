// Driver archetype bodies. Declarations are in Api/Archetypes.cppm.
//
// A **module implementation unit** — `module raceengine.traffic;` with no `export` — so editing a
// number here rebuilds one object rather than every importer of `raceengine`. That matters more for
// this file than for most: these are the numbers the seat will move.
module;

#include <string>
#include <string_view>
#include <vector>

module raceengine.traffic;

namespace raceengine
{

DriverBehaviour behaviourFor(const DriverArchetype archetype)
{
    switch (archetype)
    {
    case DriverArchetype::Cautious:
        // Ten km/h under the limit is 2.8 m/s, which is the figure the brief asks for. Everything
        // else follows from the same person: a long gap, a slow take-off, gentle pedals, and no
        // lane change unless the road forces one.
        return DriverBehaviour{.speedOffsetMetresPerSecond = -2.8,
                               .speedSpreadMetresPerSecond = 0.8,
                               .desiredHeadwaySeconds = 2.4,
                               .minimumGapMetres = 4.0,
                               .comfortableAccelerationMetresPerSecondSquared = 0.9,
                               .comfortableDecelerationMetresPerSecondSquared = 1.6,
                               .accelerationExponent = 4.0,
                               .emergencyDecelerationMetresPerSecondSquared = 7.0,
                               .greenDelaySeconds = 1.6,
                               .startDelaySeconds = 1.0,
                               .changesLanesToGetOn = false,
                               .politeness = 0.8,
                               .changeThresholdMetresPerSecondSquared = 0.6,
                               .safeDecelerationMetresPerSecondSquared = 2.5,
                               .changeCooldownSeconds = 12.0,
                               .changeDurationSeconds = 3.6,
                               .overtakeUrgeMetresPerSecondSquared = 0.0};

    case DriverArchetype::Aggressive:
        // Ten km/h over, and the rest of the person: a short gap, a quick take-off, a low regard
        // for whoever is being pulled in front of, and a standing urge to get past anything
        // slower. The safe deceleration stays high enough that they do not actually cause
        // collisions — an aggressive driver here is one who takes gaps, not one who has crashes.
        return DriverBehaviour{.speedOffsetMetresPerSecond = 2.8,
                               .speedSpreadMetresPerSecond = 1.4,
                               .desiredHeadwaySeconds = 0.7,
                               .minimumGapMetres = 1.4,
                               .comfortableAccelerationMetresPerSecondSquared = 2.6,
                               .comfortableDecelerationMetresPerSecondSquared = 3.2,
                               .accelerationExponent = 4.0,
                               .emergencyDecelerationMetresPerSecondSquared = 9.0,
                               .greenDelaySeconds = 0.15,
                               .startDelaySeconds = 0.15,
                               .changesLanesToGetOn = true,
                               .politeness = 0.05,
                               .changeThresholdMetresPerSecondSquared = 0.05,
                               .safeDecelerationMetresPerSecondSquared = 5.5,
                               .changeCooldownSeconds = 2.5,
                               .changeDurationSeconds = 1.8,
                               .overtakeUrgeMetresPerSecondSquared = 0.5};

    case DriverArchetype::Regular:
        break;
    }

    // The limit, a two-second gap and ordinary pedals. Every field the two above move is moved away
    // from one of these, which is what makes this the one to read first.
    return DriverBehaviour{.speedOffsetMetresPerSecond = 0.0,
                           .speedSpreadMetresPerSecond = 1.1,
                           .desiredHeadwaySeconds = 1.6,
                           .minimumGapMetres = 2.5,
                           .comfortableAccelerationMetresPerSecondSquared = 1.5,
                           .comfortableDecelerationMetresPerSecondSquared = 2.2,
                           .accelerationExponent = 4.0,
                           .emergencyDecelerationMetresPerSecondSquared = 8.0,
                           .greenDelaySeconds = 0.7,
                           .startDelaySeconds = 0.45,
                           .changesLanesToGetOn = true,
                           .politeness = 0.35,
                           .changeThresholdMetresPerSecondSquared = 0.25,
                           .safeDecelerationMetresPerSecondSquared = 4.0,
                           .changeCooldownSeconds = 7.0,
                           .changeDurationSeconds = 2.6,
                           .overtakeUrgeMetresPerSecondSquared = 0.1};
}

std::string_view archetypeName(const DriverArchetype archetype)
{
    switch (archetype)
    {
    case DriverArchetype::Cautious:
        return "cautious";
    case DriverArchetype::Aggressive:
        return "aggressive";
    case DriverArchetype::Regular:
        break;
    }

    return "regular";
}

std::vector<DriverProfile> defaultDriverProfiles()
{
    return std::vector<DriverProfile>{DriverProfile{.name = std::string(archetypeName(DriverArchetype::Cautious)),
                                                    .weight = 0.25,
                                                    .behaviour = behaviourFor(DriverArchetype::Cautious)},
                                      DriverProfile{.name = std::string(archetypeName(DriverArchetype::Regular)),
                                                    .weight = 0.55,
                                                    .behaviour = behaviourFor(DriverArchetype::Regular)},
                                      DriverProfile{.name = std::string(archetypeName(DriverArchetype::Aggressive)),
                                                    .weight = 0.20,
                                                    .behaviour = behaviourFor(DriverArchetype::Aggressive)}};
}

} // namespace raceengine
