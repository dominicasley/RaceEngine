// The police's radio, the search, and the route service (docs/pursuit-radio-brief.md): a route asked
// for lands later and the unit drives meanwhile; the worker answers on its own thread; a unit that
// cannot see the player routes to the general location and only a sighting moves it; and a chase
// that has lost the player fans out on different bearings and widens the ring. The same fixtures as
// PoliceTests.cpp: a straight road on a plate, built in code.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

import raceengine;

using raceengine::behaviourFor;
using raceengine::bringUpJolt;
using raceengine::buildLaneNetwork;
using raceengine::DriverArchetype;
using raceengine::DriverProfile;
using raceengine::generateProvingGround;
using raceengine::LaneSource;
using raceengine::PendingRoute;
using raceengine::PhysicsWorld;
using raceengine::ProvingGroundDescriptor;
using raceengine::PursuitDirector;
using raceengine::PursuitOptions;
using raceengine::PursuitPlayer;
using raceengine::PursuitRouter;
using raceengine::PursuitRouteWorker;
using raceengine::PursuitUnit;
using raceengine::RouteAnswer;
using raceengine::RouteRequest;
using raceengine::RouteService;
using raceengine::tearDownJolt;
using raceengine::TrafficPopulation;
using raceengine::TrafficPopulationOptions;
using raceengine::TrafficUpdate;

using Catch::Matchers::WithinAbs;

namespace
{

struct JoltGuard
{
    JoltGuard()
    {
        REQUIRE(bringUpJolt().has_value());
    }

    ~JoltGuard()
    {
        tearDownJolt();
    }

    JoltGuard(const JoltGuard&) = delete;
    JoltGuard(JoltGuard&&) = delete;
    JoltGuard& operator=(const JoltGuard&) = delete;
    JoltGuard& operator=(JoltGuard&&) = delete;
};

constexpr auto tick = 1.0 / 360.0;

[[nodiscard]] PhysicsWorld plate(const double length, const double width)
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = length;
    descriptor.width = width;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    auto ground = generateProvingGround(descriptor);
    REQUIRE(ground.has_value());

    auto world = PhysicsWorld::create(ground.value());
    REQUIRE(world.has_value());

    return std::move(world).value();
}

[[nodiscard]] raceengine::LaneNetwork straightRoad(const double length, const double limit)
{
    auto line = std::vector<glm::dvec3>();
    const auto span = length - 40.0;
    const auto count = static_cast<int>(span / 18.0);

    for (auto index = 0; index <= count; index++)
    {
        line.push_back(glm::dvec3(0.0, 0.0, 20.0 + span * static_cast<double>(index) / static_cast<double>(count)));
    }

    auto built = buildLaneNetwork({LaneSource{.id = 1,
                                              .name = "straight",
                                              .speedLimitMetresPerSecond = limit,
                                              .allowLaneChanges = false,
                                              .allowUTurns = false,
                                              .points = std::move(line)}});
    REQUIRE(built.has_value());

    return std::move(built).value();
}

// A city of patrol cars and nothing else, capped at a stated number of them.
[[nodiscard]] TrafficPopulation patrolCars(raceengine::LaneNetwork network, const std::size_t count)
{
    auto options = TrafficPopulationOptions{};
    options.densityPerKilometre = 20.0;
    options.maximumAgents = count;
    options.bodyCount = 2;
    options.policeBody = 1;
    options.policeShare = 1.0;
    options.recoverySeconds = 30.0;

    auto city = TrafficPopulation(
        std::move(network), options,
        {DriverProfile{.name = "regular", .weight = 1.0, .behaviour = behaviourFor(DriverArchetype::Regular)}});
    city.seed();

    REQUIRE(city.agents().size() == count);
    REQUIRE(city.policeIds().size() == count);

    return city;
}

[[nodiscard]] PursuitPlayer playerAt(const double z, const double speed)
{
    return PursuitPlayer{.positionMetres = glm::dvec3(0.0, 0.0, z),
                         .velocityMetresPerSecond = glm::dvec3(0.0, 0.0, speed),
                         .forward = glm::dvec3(0.0, 0.0, 1.0),
                         .lengthMetres = 4.3,
                         .widthMetres = 1.8};
}

void run(PursuitDirector& director, TrafficPopulation& city, const PhysicsWorld& world, const PursuitPlayer& player,
         const double seconds, double& clock)
{
    const auto steps = static_cast<int>(std::lround(seconds / tick));
    const auto vehicles =
        std::array{raceengine::TrafficVehicle{.positionMetres = player.positionMetres,
                                              .velocityMetresPerSecond = player.velocityMetresPerSecond,
                                              .forward = player.forward,
                                              .lengthMetres = player.lengthMetres}};

    for (auto index = 0; index < steps; index++)
    {
        city.update(TrafficUpdate{.deltaTimeSeconds = tick,
                                  .simulatedSeconds = clock,
                                  .focusMetres = player.positionMetres,
                                  .vehicles = vehicles},
                    world);
        director.update(tick, player, city, world);
        clock += tick;
    }
}

// A point on the road at least `clearance` from every patrol car, or as far as the road allows.
[[nodiscard]] double roadPointClearOf(const TrafficPopulation& city, const double length, const double clearance)
{
    auto bestZ = 30.0;
    auto bestGap = -1.0;

    for (auto z = 30.0; z < length - 30.0; z += 2.0)
    {
        auto gap = 1.0e9;
        for (const auto& agent : city.agents())
        {
            gap = std::min(gap, std::abs(agent.positionMetres.z - z));
        }

        if (gap > bestGap)
        {
            bestGap = gap;
            bestZ = z;
        }
    }

    REQUIRE(bestGap >= clearance);

    return bestZ;
}

// A service that answers each request a stated number of collects after it was asked, on the
// calling thread — the worker's lag with none of its timing — and, when told to, answers the first
// request under the wrong serial.
class DelayedService final : public RouteService
{
public:
    DelayedService(PursuitRouter& way, const int delayCollects, const bool corruptFirst) :
        router(&way),
        delay(delayCollects),
        corrupt(corruptFirst)
    {
    }

    void request(const std::uint32_t agent, const std::uint64_t serial, const RouteRequest& request) override
    {
        held.push_back(Held{.pending = PendingRoute{.agent = agent, .serial = serial, .request = request}, .collects = 0});
        asked++;
    }

    void collect(std::vector<RouteAnswer>& into) override
    {
        for (auto index = std::size_t{0}; index < held.size();)
        {
            auto& entry = held[index];
            entry.collects++;

            if (entry.collects < delay)
            {
                index++;

                continue;
            }

            auto serial = entry.pending.serial;
            if (corrupt && answered == 0)
            {
                serial = serial - 1;
            }

            into.push_back(RouteAnswer{
                .agent = entry.pending.agent, .serial = serial, .route = router->route(entry.pending.request)});
            answered++;
            held.erase(held.begin() + static_cast<std::ptrdiff_t>(index));
        }
    }

    [[nodiscard]] std::size_t pending() const override
    {
        return held.size();
    }

    [[nodiscard]] std::size_t searches() const override
    {
        return answered;
    }

    [[nodiscard]] double worstSearchSeconds() const override
    {
        return 0.0;
    }

    std::size_t asked = 0;
    std::size_t answered = 0;

private:
    struct Held
    {
        PendingRoute pending;
        int collects;
    };

    PursuitRouter* router;
    int delay;
    bool corrupt;
    std::vector<Held> held;
};

} // namespace

TEST_CASE("the worker answers route requests on its own thread, one per unit, and stops clean",
          "[police][police-worker]")
{
    auto city = patrolCars(straightRoad(900.0, 14.0), 1);

    auto worker = PursuitRouteWorker(PursuitRouter(city.network(), nullptr));

    worker.request(7, 1, RouteRequest{.fromMetres = glm::dvec3(0.0, 0.0, 50.0), .toMetres = glm::dvec3(0.0, 0.0, 500.0)});
    worker.request(8, 1, RouteRequest{.fromMetres = glm::dvec3(0.0, 0.0, 700.0), .toMetres = glm::dvec3(0.0, 0.0, 100.0)});

    auto answers = std::vector<RouteAnswer>();
    const auto started = std::chrono::steady_clock::now();

    while (answers.size() < 2 && std::chrono::steady_clock::now() - started < std::chrono::seconds(5))
    {
        worker.collect(answers);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    REQUIRE(answers.size() == 2);
    REQUIRE(worker.pending() == 0);
    REQUIRE(worker.searches() == 2);

    for (const auto& answer : answers)
    {
        REQUIRE((answer.agent == 7 || answer.agent == 8));
        REQUIRE(answer.serial == 1);
        REQUIRE(answer.route.found);
        REQUIRE(answer.route.points.size() >= 2);
        REQUIRE(answer.route.searchSeconds >= 0.0);
    }

    // Two asks for one unit in a row: the later serial is answered; the earlier is answered only if
    // the thread had already taken it.
    worker.request(9, 5, RouteRequest{.fromMetres = glm::dvec3(0.0, 0.0, 50.0), .toMetres = glm::dvec3(0.0, 0.0, 300.0)});
    worker.request(9, 6, RouteRequest{.fromMetres = glm::dvec3(0.0, 0.0, 50.0), .toMetres = glm::dvec3(0.0, 0.0, 400.0)});

    answers.clear();
    const auto again = std::chrono::steady_clock::now();
    auto sawNewest = false;

    while (!sawNewest && std::chrono::steady_clock::now() - again < std::chrono::seconds(5))
    {
        worker.collect(answers);
        for (const auto& answer : answers)
        {
            REQUIRE(answer.agent == 9);
            REQUIRE((answer.serial == 5 || answer.serial == 6));
            sawNewest = sawNewest || answer.serial == 6;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    REQUIRE(sawNewest);
    // Destroyed with nothing waiting, and with something waiting: both join.
    worker.request(10, 1, RouteRequest{.fromMetres = glm::dvec3(0.0, 0.0, 50.0), .toMetres = glm::dvec3(0.0, 0.0, 400.0)});
}

TEST_CASE("a route asked for lands later, the unit drives at its goal meanwhile, and an answer under the wrong serial "
          "is dropped",
          "[police][police-radio]")
{
    const JoltGuard jolt;

    const auto world = plate(900.0, 60.0);
    auto city = patrolCars(straightRoad(900.0, 14.0), 1);
    auto router = PursuitRouter(city.network(), nullptr);

    auto options = PursuitOptions{};
    options.sightRadiusMetres = 400.0;

    SECTION("the answer lands after the delay")
    {
        auto service = DelayedService(router, 108, false);
        auto director = PursuitDirector{options};
        director.setRouteService(&service);
        auto clock = 0.0;

        const auto& patrol = city.agents().front();
        const auto player = playerAt(patrol.positionMetres.z + 250.0, 28.0);

        // A quarter of a second in: the chase is on, the unit is routing, and its route is out — the
        // service answers three tenths after the ask.
        run(director, city, world, player, 0.25, clock);
        REQUIRE(director.status().active);
        REQUIRE(director.units().size() == 1);

        const auto& asked = director.units().front();
        REQUIRE(asked.routing);
        REQUIRE(asked.nearPlayer);
        REQUIRE(asked.routePending);
        REQUIRE(asked.route.points.size() < 2);
        REQUIRE(service.asked == 1);
        REQUIRE(director.status().routesRequested == 1);
        REQUIRE(director.status().routesBuilt == 0);
        // Meanwhile it drives at the player: lead pursuit, a second ahead.
        REQUIRE_THAT(asked.aimMetres.z, WithinAbs(player.positionMetres.z + 28.0, 1e-6));

        // A tenth of a second later the answer is in and the aim is on the road ahead of the unit.
        run(director, city, world, player, 0.1, clock);
        const auto& landed = director.units().front();
        REQUIRE_FALSE(landed.routePending);
        REQUIRE(landed.route.found);
        REQUIRE(landed.route.points.size() >= 2);
        REQUIRE(director.status().routesBuilt == 1);
        REQUIRE(landed.aimMetres.z > city.agents().front().positionMetres.z);
        REQUIRE(landed.aimMetres.z < player.positionMetres.z);
    }

    SECTION("an answer under the wrong serial is dropped and the unit keeps waiting")
    {
        auto service = DelayedService(router, 2, true);
        auto director = PursuitDirector{options};
        director.setRouteService(&service);
        auto clock = 0.0;

        const auto& patrol = city.agents().front();
        const auto player = playerAt(patrol.positionMetres.z + 250.0, 28.0);

        run(director, city, world, player, 0.5, clock);
        REQUIRE(director.status().active);
        REQUIRE(director.units().size() == 1);

        const auto& unit = director.units().front();
        REQUIRE(unit.routing);
        REQUIRE(unit.routePending);
        REQUIRE(unit.route.points.size() < 2);
        REQUIRE(service.answered == 1);
        REQUIRE(director.status().routesBuilt == 0);
        // And it did not ask again while the answer was out.
        REQUIRE(service.asked == 1);
    }
}

TEST_CASE("the radio: a unit that cannot see the player routes to the general location, which only a sighting moves",
          "[police][police-radio]")
{
    const JoltGuard jolt;

    const auto world = plate(900.0, 60.0);
    auto city = patrolCars(straightRoad(900.0, 14.0), 1);
    auto router = PursuitRouter(city.network(), nullptr);

    auto director = PursuitDirector{};
    director.setRouter(&router);
    auto clock = 0.0;

    const auto& patrol = city.agents().front();
    const auto c = patrol.positionMetres.z;

    // Seen thirty metres up the road, speeding: the chase starts and the report is the player.
    const auto seen = playerAt(c + 30.0, 28.0);
    run(director, city, world, seen, 0.3, clock);
    REQUIRE(director.status().active);
    REQUIRE(director.status().observed);
    REQUIRE(director.status().broadcasts == 1);
    REQUIRE(director.units().size() == 1);

    // Seventy metres further and still in sight: the report has moved past the radio's threshold and
    // the general location is re-broadcast.
    const auto further = playerAt(c + 100.0, 28.0);
    run(director, city, world, further, 0.3, clock);
    REQUIRE(director.status().observed);
    REQUIRE(director.status().broadcasts == 2);
    REQUIRE(director.units().front().nearPlayer);
    REQUIRE_THAT(director.units().front().goalMetres.z, WithinAbs(c + 100.0, 1e-6));

    // Four hundred metres away, out of sight: the unit is far, its goal is the last broadcast and not
    // the player, and the report ages.
    const auto gone = playerAt(c + 400.0, 28.0);
    run(director, city, world, gone, 0.4, clock);
    REQUIRE_FALSE(director.status().observed);
    REQUIRE(director.status().broadcasts == 2);
    REQUIRE(director.status().reportAgeSeconds > 0.3);
    REQUIRE(director.status().farUnits == 1);
    REQUIRE(director.status().nearUnits == 0);
    const auto& far = director.units().front();
    REQUIRE_FALSE(far.nearPlayer);
    REQUIRE(far.routing);
    REQUIRE_THAT(far.goalMetres.z, WithinAbs(c + 100.0, 1e-6));
    REQUIRE(std::abs(far.aimMetres.z - (c + 400.0)) > 100.0);
    // Not yet searching: a corner is not a loss.
    REQUIRE_FALSE(director.status().searching);

    // Back in sight, twenty metres up the road: the report is the player again, at once.
    const auto back = playerAt(c + 20.0, 28.0);
    run(director, city, world, back, 0.3, clock);
    REQUIRE(director.status().observed);
    REQUIRE_THAT(director.status().reportAgeSeconds, WithinAbs(0.0, 1e-9));
    REQUIRE(director.units().front().nearPlayer);
}

TEST_CASE("the radio: a unit near the player routes to the player itself even when it cannot see it",
          "[police][police-radio]")
{
    const JoltGuard jolt;

    const auto world = plate(900.0, 60.0);
    auto city = patrolCars(straightRoad(900.0, 14.0), 1);
    auto router = PursuitRouter(city.network(), nullptr);

    // Short sight, so a player fifty metres off is unseen and still inside the radio's near distance.
    auto options = PursuitOptions{};
    options.sightRadiusMetres = 40.0;

    auto director = PursuitDirector{options};
    director.setRouter(&router);
    auto clock = 0.0;

    const auto& patrol = city.agents().front();
    const auto c = patrol.positionMetres.z;

    run(director, city, world, playerAt(c + 30.0, 28.0), 0.3, clock);
    REQUIRE(director.status().active);
    REQUIRE(director.status().observed);

    const auto unseenNear = playerAt(c + 52.0, 28.0);
    run(director, city, world, unseenNear, 0.3, clock);
    REQUIRE_FALSE(director.status().observed);
    REQUIRE(director.status().nearUnits == 1);

    const auto& unit = director.units().front();
    REQUIRE(unit.nearPlayer);
    REQUIRE_FALSE(unit.sighted);
    REQUIRE(unit.routing);
    REQUIRE_THAT(unit.goalMetres.z, WithinAbs(c + 52.0, 1e-6));
}

TEST_CASE("the search: units that have lost the player fan out on different bearings from the last sighting, the ring "
          "widens, and the chase outlasts the old fifteen seconds",
          "[police][police-search]")
{
    const JoltGuard jolt;

    // A wide plate: the hidden player stands well off the road, beyond the radio's near distance
    // from the lane, or a searcher passing on the lane homes in on it and finds it — which the
    // first run of this fixture did, at 25 m off the road, exactly as the near rule says.
    constexpr auto length = 900.0;
    const auto world = plate(length, 240.0);
    auto city = patrolCars(straightRoad(length, 14.0), 3);
    auto router = PursuitRouter(city.network(), nullptr);

    // Short sight, and everybody joins whatever the distance, so three units can lose one player on
    // a straight road.
    auto options = PursuitOptions{};
    options.sightRadiusMetres = 20.0;
    options.joinRadiusMetres = 2000.0;

    auto director = PursuitDirector{options};
    director.setRouter(&router);
    auto clock = 0.0;

    // Seen by the patrol car with the most road either side of it, so the ring behind the sighting
    // is on the road too.
    const auto* seer = &city.agents().front();
    for (const auto& agent : city.agents())
    {
        const auto room = std::min(agent.positionMetres.z - 20.0, length - 20.0 - agent.positionMetres.z);
        const auto seerRoom = std::min(seer->positionMetres.z - 20.0, length - 20.0 - seer->positionMetres.z);
        if (room > seerRoom)
        {
            seer = &agent;
        }
    }
    REQUIRE(seer->positionMetres.z > 120.0);

    const auto seenAt = playerAt(seer->positionMetres.z + 20.0, 28.0);
    run(director, city, world, seenAt, 0.5, clock);
    REQUIRE(director.status().active);
    REQUIRE(director.units().size() == 3);

    // The player where no patrol car can see it — clear of every car along the road and 90 m off it,
    // beyond a searcher's sight and the near distance from the lane — headed up the road as before.
    auto hidden = playerAt(roadPointClearOf(city, length, 60.0), 28.0);
    hidden.positionMetres.x = 90.0;
    REQUIRE(hidden.positionMetres.x > director.settings().radioNearMetres);
    // Six and a half seconds: the search waits six (docs/police-driving-brief.md §7) and the radio
    // reckons the runner on meanwhile, so the search is centred where the player would be by now —
    // its speed times the wait up the lane from where it was seen — and not where it was.
    run(director, city, world, hidden, 6.5, clock);

    const auto& status = director.status();
    REQUIRE(status.active);
    REQUIRE_FALSE(status.observed);
    REQUIRE(status.searching);
    REQUIRE(status.searchingUnits == 3);
    REQUIRE(status.searchSeconds > 0.4);
    REQUIRE_THAT(status.searchCentreMetres.z,
                 WithinAbs(seenAt.positionMetres.z + 28.0 * director.settings().searchAfterSeconds, 1.5));
    REQUIRE_THAT(status.searchRadiusMetres, WithinAbs(director.settings().searchRadiusMetres +
                                                          director.settings().searchGrowthMetresPerSecond * status.searchSeconds,
                                                      1e-6));

    // Three bearings, all different and all in the front half of the compass: ahead and the two
    // sides — the back is dealt only once the front nine are taken. Every goal is on the road.
    auto bearings = std::vector<double>();
    auto goals = std::vector<double>();
    auto ahead = false;
    auto back = false;

    for (const auto& unit : director.units())
    {
        REQUIRE(unit.searching);
        REQUIRE(unit.searchBearingDealt);
        REQUIRE(unit.searchGoalSet);
        REQUIRE(unit.routing);
        REQUIRE(std::abs(unit.searchGoalMetres.x) < 1.0);
        REQUIRE_THAT(unit.goalMetres.z, WithinAbs(unit.searchGoalMetres.z, 1e-6));
        REQUIRE(unit.wantedSpeedMetresPerSecond <= director.settings().searchSpeedMetresPerSecond + 1e-9);

        for (const auto other : bearings)
        {
            REQUIRE(std::abs(other - unit.searchBearingRadians) > 0.1);
        }
        REQUIRE(std::cos(unit.searchBearingRadians) >= -1e-9);
        bearings.push_back(unit.searchBearingRadians);
        goals.push_back(unit.searchGoalMetres.z);

        const auto offset = unit.searchGoalMetres.z - status.searchCentreMetres.z;
        ahead = ahead || offset > 40.0;
        back = back || offset < -40.0;
    }

    REQUIRE(ahead);
    REQUIRE_FALSE(back);

    // Twenty seconds in, still searching and the chase still on — the old rule would have called it
    // off at fifteen — and the ring has grown with the clock.
    run(director, city, world, hidden, 18.0, clock);
    REQUIRE(director.status().active);
    REQUIRE(director.status().searching);
    REQUIRE(director.status().searchRadiusMetres > 140.0);

    // The unit sent on ahead has reached its first goal and taken the next, a step further out.
    auto stepped = false;
    for (const auto& unit : director.units())
    {
        stepped = stepped || unit.searchRadiusMetres >= director.settings().searchRadiusMetres +
                                                             director.settings().searchStepMetres - 1e-6;
    }
    REQUIRE(stepped);

    // A sighting ends the search at once: the player in front of whichever unit is nearest.
    const auto* nearest = &director.units().front();
    for (const auto& unit : director.units())
    {
        if (unit.distanceMetres < nearest->distanceMetres)
        {
            nearest = &unit;
        }
    }
    // Three metres ahead of it and on its line, so it is inside the twenty metres whichever way that
    // unit is going and however far it gets in two sight ticks.
    const auto& seenBy = city.agents()[nearest->agent];
    auto found = playerAt(seenBy.positionMetres.z + 3.0, 28.0);
    found.positionMetres.x = seenBy.positionMetres.x;
    run(director, city, world, found, 0.2, clock);
    REQUIRE(director.status().observed);
    REQUIRE_FALSE(director.status().searching);
    REQUIRE(director.status().active);
}

TEST_CASE("the search: a chase that never sees the player again is called off at the new limit and not before",
          "[police][police-search]")
{
    const JoltGuard jolt;

    constexpr auto length = 900.0;
    const auto world = plate(length, 60.0);
    auto city = patrolCars(straightRoad(length, 14.0), 1);
    auto router = PursuitRouter(city.network(), nullptr);

    auto options = PursuitOptions{};
    options.sightRadiusMetres = 30.0;
    options.searchSpeedMetresPerSecond = 0.0;

    auto director = PursuitDirector{options};
    director.setRouter(&router);
    auto clock = 0.0;

    const auto& first = city.agents().front();
    run(director, city, world, playerAt(first.positionMetres.z + 20.0, 28.0), 0.5, clock);
    REQUIRE(director.status().active);

    // Hidden, and the searching unit held still by the search speed, so it never stumbles on the
    // player: on at forty, off at forty-six.
    const auto hidden = playerAt(roadPointClearOf(city, length, 60.0), 0.0);
    run(director, city, world, hidden, 40.0, clock);
    REQUIRE(director.status().active);
    REQUIRE(director.status().searching);

    run(director, city, world, hidden, 6.0, clock);
    REQUIRE_FALSE(director.status().active);
    REQUIRE_FALSE(director.status().searching);
}
