module;

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module raceengine.traffic:Pursuit;

import raceengine.physics;

import :Lanes;
import :Population;
import :PursuitPlan;
import :PursuitRoute;

namespace raceengine
{

// The police: what a patrol car can see, what it counts as an offence, how a chase starts, who joins
// it, how the cars work together to stop the player, and when it is over (docs/police-pursuit-brief.md).
//
// **This is the law and the tactics, and it steps nothing.** The bodies the patrol cars drive as
// belong to `TrafficPopulation` (the cheap tier, `AgentMode::Pursuing`) or to the caller (the full
// vehicle model, `AgentMode::External`); this class reads their poses, decides where each should be
// and how fast, and writes that back as an aim. Which car gets the full model is the caller's call —
// the caller owns the vehicle model and the budget — and it says so with `setFullModel`. Keeping the
// decisions here and the stepping elsewhere is what lets every rule below be tested on a lane
// network built in a fixture with no vehicle model in the room.
//
// Metres, seconds, radians, world frame — the frame the population's poses are in.

// What the player was seen doing.
export enum class Offence : std::uint8_t
{
    None,
    Speeding,
    OffRoad,
    WrongWay,
    HitTraffic,
    HitPolice
};

export [[nodiscard]] const char* offenceName(Offence offence);

// A unit's job in the chase. The stations are in the player's own frame — behind, ahead, either
// side — and a unit too far away to hold one simply chases.
export enum class PursuitRole : std::uint8_t
{
    Chase,
    Tail,
    Lead,
    FlankLeft,
    FlankRight
};

export [[nodiscard]] const char* pursuitRoleName(PursuitRole role);

export struct PursuitOptions
{
    // --- vision --------------------------------------------------------------------------------

    // How far a patrol car can see the player. A city block and a half.
    double sightRadiusMetres = 110.0;
    // The cosine of the half-angle a cruising patrol car looks through, about its own heading. A
    // little past a right angle either side: a patrol car has mirrors. A car already in the chase
    // looks everywhere.
    double sightConeCosine = -0.2;
    // Eye heights for the line-of-sight ray, metres above each car's origin.
    double eyeHeightMetres = 1.2;
    double targetHeightMetres = 0.9;
    // How often the sight rays are cast. Ten times a second is finer than a person notices a car.
    double sightIntervalSeconds = 0.1;

    // --- what counts -----------------------------------------------------------------------------

    // How far over the lane's limit the player may be before it is speeding, m/s. Three is about
    // 11 km/h, which is where a real patrol car stops giving the benefit of the doubt.
    double speedingToleranceMetresPerSecond = 3.0;
    // How far from the nearest lane the player has to be before it is off the road. Wider than a
    // lane's own width by a margin, so a car using the whole carriageway is not.
    double offRoadLateralMetres = 7.0;
    // Below this ground speed nothing is an offence: a car being parked is not driving the wrong way.
    double offenceSpeedFloorMetresPerSecond = 2.5;
    // Driving against the lane at this speed along it, m/s, is the wrong way.
    double wrongWaySpeedMetresPerSecond = 2.5;
    // A hit on the bodywork under this velocity change, m/s, is a brush and not a collision.
    double collisionThresholdMetresPerSecond = 1.0;

    // --- the felony meter, 0 to 1 --------------------------------------------------------------

    // Per second, while the offence is being watched. Speeding scales with the overspeed on top.
    double felonyPerSecondSpeeding = 0.020;
    double felonyPerSecondOffRoad = 0.030;
    double felonyPerSecondWrongWay = 0.050;
    // Per collision, watched. Police always know when they have been hit.
    double felonyPerTrafficHit = 0.10;
    double felonyPerPoliceHit = 0.15;
    // Per second of being chased in sight.
    double felonyPerSecondEvading = 0.012;
    // Per second, once nobody can see the player and nobody is chasing.
    double felonyDecayPerSecond = 0.010;
    // Where every patrol car on the map joins whatever the distance.
    double swarmFelony = 1.0;

    // --- the chase -----------------------------------------------------------------------------

    // A patrol car this close to the player joins a chase that is already on.
    double joinRadiusMetres = 300.0;
    // How many cars may chase at once below the swarm.
    std::size_t maximumUnits = 6;
    // How long the player has to stay out of every unit's sight for the chase to be called off.
    // Forty-five since 2026-09-10, from fifteen: the units search for the player now
    // (docs/pursuit-radio-brief.md, §1.3), and a search that is called off before the far units reach
    // the place the player was last seen is not one. A gameplay number, placed.
    double loseSightSeconds = 45.0;
    // The fastest a unit is asked to go, m/s, and how much faster than the player a unit may run to
    // close on its station. Seventy since 2026-09-12 (docs/police-driving-brief.md §9), from
    // sixty-two: the Golf's own top speed is about sixty-nine, and a cap under it is a straight the
    // runner wins by construction.
    double maximumUnitSpeedMetresPerSecond = 70.0;
    double closingMarginMetresPerSecond = 10.0;
    // Past this distance from its station a unit stops holding formation and chases.
    double chaseDistanceMetres = 45.0;
    // How often the roles are dealt out again.
    double roleIntervalSeconds = 1.5;
    // A unit holding a station steers at a point on the station's own line — the line through the
    // station along the player's heading — this far ahead of itself, at least the first and this
    // many seconds of its own speed otherwise. The along-track error is the speed's to close and
    // never the wheel's: steering at the station itself asked a car abreast of its flank station
    // for a right-angle turn, and the cornering cap then held every flank to walking pace
    // (2026-09-09, the second seat).
    double stationLookAheadMetres = 10.0;
    double stationLookAheadSeconds = 1.2;
    // Above this speed a car facing away from its aim stops in a straight line before it turns
    // round; below it, full lock and a gentle throttle turn it in the road.
    double turnAroundSpeedMetresPerSecond = 6.0;

    // --- backing up ----------------------------------------------------------------------------

    // A unit asked to move that stays under this speed for this long is stuck against something —
    // the player, a kerb, a wall, another unit — and backs up for this long before trying again. A
    // unit slower than walking pace whose station is behind it and inside this distance backs into
    // it rather than sitting on the brakes. Both back up at this speed.
    double stuckSpeedMetresPerSecond = 0.6;
    double stuckSeconds = 0.8;
    double reverseSeconds = 1.6;
    double reverseAimMetres = 9.0;
    double reverseSpeedMetresPerSecond = 4.0;

    // --- navigation (docs/pursuit-navigation-brief.md, stages 2 and 3) ----------------------------

    // The line of drive: one more ray a sight tick, from the unit's origin to the player's at this
    // height — over a kerb and under nothing a car fits under. Blocked short of the player by more
    // than the clearance, or the player further than `chaseDistanceMetres`, and the unit routes.
    // Ten centimetres since 2026-09-12, from sixty (docs/police-driving-brief.md §10): at sixty the ray
    // cleared a kerb by design, and a unit inside reach drove the straight line at the player over
    // every kerb between them; at ten a kerb's face blocks it and the unit routes, on the road. A
    // hit whose normal is mostly up — the road on a rise — does not block it. The corridor and the
    // room rays keep their own height, which clears a kerb, because the route they follow crosses
    // kerbs where the mesh says it may.
    double lineOfDriveHeightMetres = 0.10;
    double lineOfDriveClearanceMetres = 3.0;
    double corridorHeightMetres = 0.6;
    // A routing unit stays on its route until its line has been clear, inside reach, for this long,
    // so a wall's end does not flick it between the two every tick.
    double routeHoldSeconds = 0.5;
    // How often a unit's route is rebuilt — one route a tick at most across the roster — and how far
    // off its route a unit may drift before it is rebuilt at once.
    double routeIntervalSeconds = 0.5;
    double routeDriftMetres = 15.0;
    // Where the aim sits on the route: a look-ahead ahead of the unit's place on it, the drivers' own
    // (the larger of the metres and the seconds of speed); and how far ahead the wanted speed is read.
    double routeLookAheadMetres = 6.0;
    double routeLookAheadSeconds = 0.55;
    double reactionSeconds = 0.3;
    // The line through a bend (docs/police-driving-brief.md §12). A routing unit's aim is the
    // route's own point a look-ahead on, pure pursuit carrying the arc; no curvature is fed forward
    // (the profile's estimate spreads eight metres either side of a corner, and a wheel fed it took
    // an early apex and ran wide at the exit). The route is the lane's chord polyline, control
    // points this far apart, which sits inside the true arc by the chords' sagitta: the aim is
    // pushed back out by the mean of it. And the apex is cut inside the lane's centre by this much —
    // with the cut pure pursuit makes on its own, about the 0.6 that leaves the 0.3 m Dominic asked
    // for between a wheel and a kerb at a 3.7 m lane's edge — in full at bends tighter than the
    // tight curvature (a 33 m radius), not at all at bends gentler than the gentle one (200 m), and
    // tapering back to nothing toward a hairpin, where the natural cut is larger.
    double routeChordMetres = 8.0;
    double apexInsideMetres = 0.3;
    double apexTightCurvature = 0.03;
    double apexGentleCurvature = 0.005;
    double apexHairpinCurvature = 0.08;
    // The profile. The share of the tyre a corner is planned at; the ground's share of the road's
    // grip; the full model's cornering limit and straight-line deceleration — the second **measured**
    // on 2026-09-10 (`[police-brake]`: the Charger from 30 to 5 m/s on the plate, the pedal on the
    // floor, its anti-lock on, 9.30 m/s² mean; 8 was the placeholder); the cheap body's, its 0.85 μ g
    // and min(brake / m, μ g); and the acceleration the forward pass allows.
    double cornerMargin = 0.9;
    double offRoadGripFactor = 0.7;
    double fullModelCorneringLimitG = 0.85;
    double fullModelBrakingMetresPerSecondSquared = 9.3;
    double cheapBodyCorneringLimitG = 0.85 * 1.10;
    double cheapBodyBrakingMetresPerSecondSquared = 10.3;
    // Eight since 2026-09-12, from four (docs/police-driving-brief.md §9): the forward pass is what
    // the wanted speed climbs at out of a corner, and the driver's throttle follows the wanted speed,
    // so four held every unit to half what the car could do out of every corner.
    double accelerationMetresPerSecondSquared = 8.0;
    // The full model's driver caps the speed for a corner in the **path** — the route's curvature a
    // look-ahead ahead — and, off a route, only when the aim is more than this far off the nose: a
    // small lateral error at speed is corrected by the wheel, at the curvature the tyre holds at that
    // speed, and never by slowing down. Before this the cap read the pure-pursuit demand for a
    // two-metre error at forty metres a second as a corner and held the unit to thirty-two.
    double steeringCapAngleRadians = 0.35;

    // --- the radio and the search (docs/pursuit-radio-brief.md) ----------------------------------
    //
    // A unit that sees the player, or is inside this distance of it, is **near**: it routes to the
    // player itself. Every other unit is **far** and routes to the general location — the player as
    // last called in by any patrol car that saw it — which is re-broadcast only when the report has
    // moved this far from the last broadcast, so a far unit's route is rebuilt on the broadcast and
    // not on the clock.
    double radioNearMetres = 60.0;
    double radioUpdateMetres = 30.0;
    // The report is **dead-reckoned** (docs/police-driving-brief.md §7): the general location the
    // far units are sent to is the player as last seen, advanced along its lane at the speed it had
    // for as long as it has been unseen, up to this long — a runner that went round a corner is
    // still going, and a far unit that drives to where it *was* arrives at an empty road. Thirty
    // metres, from sixty, because the point now moves with the clock and a route rebuilt every
    // thirty metres of it is what keeps the far units on the runner's heels.
    double deadReckonSeconds = 8.0;
    // Nobody has seen the player for this long and the units search: each on its own bearing from
    // the place the player was reckoned to be, to a goal on the ring whose radius starts here and
    // grows at this rate; a unit within the arrival distance of its goal takes the next one a step
    // further out on its bearing. Goals are put on the nearest lane within the snap distance.
    // Searching units run at no more than the search speed, and a profile to a search goal ends at
    // the goal speed — slow enough to look. Six seconds, from a second and a half: in a city the
    // sight lines break at every corner, and a swarm that fanned out backwards and sideways at every
    // break is what Dominic found turning round in the road (§7). The front half of the compass is
    // dealt before the back half.
    double searchAfterSeconds = 6.0;
    double searchRadiusMetres = 60.0;
    double searchGrowthMetresPerSecond = 5.0;
    double searchArriveMetres = 15.0;
    double searchStepMetres = 70.0;
    double searchSnapMetres = 80.0;
    double searchSpeedMetresPerSecond = 40.0;
    double searchGoalSpeedMetresPerSecond = 15.0;

    // The fastest a unit threads a seam between two lanes of yielded traffic.
    double seamSpeedMetresPerSecond = 25.0;

    // --- the PIT (docs/police-driving-brief.md §10) ---------------------------------------------
    //
    // From the ram level the tail does not sit behind the player: it comes alongside the player's
    // rear quarter — this fraction of the player's length behind its centre, on the side it is
    // already on, its side this far from the player's — and once it is there (inside these along and
    // across errors, within this much of the player's pace) it steers through the quarter by this
    // much at this much over the player's speed, and the two bodies' contact does the rest.
    double pitAlongFraction = 0.3;
    double pitLateralGapMetres = 0.3;
    double pitReadyAlongMetres = 1.5;
    double pitReadyAcrossMetres = 0.8;
    double pitReadySpeedMetresPerSecond = 4.0;
    double pitPushMetres = 2.5;
    double pitSpeedMarginMetresPerSecond = 2.0;
    // The side is chosen once, when the tail first goes for the quarter, and held: chosen every tick
    // from which side of the player the unit was on, it flipped as the unit crossed the centreline
    // and the unit swerved from one station to the other down a straight (Dominic, 2026-09-12: "the
    // cars swerve side to side and take themselves out even on a straight empty wide road"). A strike
    // lasts this long at most, and the tail then drops back to its plain station for this long
    // before it goes for the quarter again.
    double pitStrikeSeconds = 1.5;
    double pitCooldownSeconds = 4.0;
    // A unit holding a station keeps it at a re-deal until it is this far onto the wrong side of the
    // player for it — a flank hovering on the centreline was dealt the other flank every second and
    // a half, and its aim swapped sides with it.
    double roleHoldMarginMetres = 1.5;

    // --- the ram (docs/police-driving-brief.md §12) ---------------------------------------------
    //
    // From the ram level a unit inside this range with the player in front of it — the bearing
    // within this cosine of its nose — drives into the player at the maximum, whatever role it holds
    // and whatever the player is doing, in place of a station: a stationary player was approached at
    // walking pace and boxed, and a unit facing it was turned round by its station line. The tail on
    // a moving player keeps the PIT. Dominic: "police are still not trying to ram into the player as
    // hard as they can this is their number on goal".
    double ramRangeMetres = 60.0;
    double ramFacingCosine = 0.5;
    double ramLeadSeconds = 0.5;

    // --- the block (docs/police-driving-brief.md §12) -------------------------------------------
    //
    // A unit ahead of the player, with the player coming at it inside this many seconds, that is not
    // lined up with the player's direction either way — coming out of a side street, round a corner
    // — does not drive on and turn round after the pass: it pulls across the player's lane and stops
    // broadside, this far along its own line and this far toward the player's centreline, and holds
    // the block this long or until the player has passed or stopped. Head on and lined up it rams
    // instead (§10). Dominic: "the correct manuever is for them to pull out and try and get tboned by
    // the player or to block the road".
    double blockSeconds = 5.0;
    double blockRunMetres = 6.0;
    double blockAcrossMetres = 4.0;
    double blockHoldSeconds = 8.0;
    double blockAlignment = 0.7;

    // --- turning round (docs/police-driving-brief.md §10) ---------------------------------------
    //
    // A unit whose aim is behind it measures the room to either side first — one ray each way at
    // the line-of-drive height, out to this reach, on the sight tick. The full-lock arc needs twice
    // this radius plus the body's width; where it fits the driver's own turn-round does it, where it
    // does not the unit does a three-point turn: forward on full lock toward the turn side until the
    // nose has used the room, back on the opposite lock through this much more heading, forward
    // again — each leg timed by the heading turned and capped at this many seconds.
    double turnRadiusMetres = 6.1;
    double turnBodyWidthMetres = 1.9;
    double turnRoomReachMetres = 16.0;
    double turnReverseRadians = 0.9;
    double turnPhaseSeconds = 4.0;

    // --- the corridor (docs/police-driving-brief.md §2.1) -------------------------------------------
    //
    // What is in front of the unit, read on the sight tick. The world: three rays at the line-of-drive
    // height — the centre and this far either side of it — pointed at the unit's aim, out to its
    // stopping distance at this share of its braking plus a reaction, floored; a hit whose normal is
    // more than this much up is the road on a rise and not an obstacle. A hit inside this many seconds
    // of the unit's speed (floored the same) blocks the corridor, which forces a route the way a
    // blocked line of drive does; any hit inside the reach caps the speed to stop this far short of
    // it. The traffic: the occupant lists, read through `TrafficPopulation::trafficAround` for the lane
    // under the unit (this far off its centre at most) and its neighbours; the speed is capped to the
    // safe approach to the car ahead at this standstill, and a chasing or routing unit shifts its aim
    // onto the neighbour whose safe speed beats its own lane's by the gain, when the car behind on that
    // neighbour is further back than this many seconds of its closing. All placed.
    double corridorHalfWidthMetres = 1.1;
    double corridorMinimumMetres = 15.0;
    double corridorRouteSeconds = 2.0;
    double corridorGroundNormalY = 0.5;
    double corridorBrakeShare = 0.7;
    double corridorStopMarginMetres = 2.0;
    double corridorLateralMetres = 7.0;
    double followStandstillMetres = 3.0;
    double mergeFollowerSeconds = 1.5;
    double laneShiftGainMetresPerSecond = 3.0;

    // --- boxing the player in ------------------------------------------------------------------

    // Bumper-to-bumper gaps the stations are placed at while the player is moving, metres, and the
    // lateral gap between the flank's side and the player's.
    double tailGapMetres = 5.0;
    double leadGapMetres = 7.0;
    double flankGapMetres = 1.6;
    // Below this player speed the box closes: the gaps shrink to touching.
    double boxSpeedMetresPerSecond = 6.0;
    double boxedTailGapMetres = 0.6;
    double boxedLeadGapMetres = 0.8;
    double boxedFlankGapMetres = 0.3;
    // From this felony level the tail unit goes for the player's rear quarter (the PIT above) rather
    // than sitting behind it. One since 2026-09-12 (docs/police-driving-brief.md §10, Dominic:
    // "wrecking is the primary goal for police. so aggressiveness should be high"); four before.
    int ramLevel = 1;

    // --- the arrest ----------------------------------------------------------------------------

    double arrestSpeedMetresPerSecond = 1.0;
    double arrestRadiusMetres = 9.0;
    double arrestSeconds = 3.0;

    // --- damage --------------------------------------------------------------------------------

    // Damage per metre per second of velocity change the bodywork takes past the collision
    // threshold, for the player and for a patrol car alike. One over twenty: a 21 m/s hit is a
    // wreck in one, a 6 m/s hit a quarter of one.
    double damagePerImpactSpeed = 1.0 / 20.0;
};

export struct PursuitUnit
{
    std::uint32_t agent = noAgent;
    PursuitRole role = PursuitRole::Chase;
    // 0 to 1. At 1 the car is a wreck and leaves the chase.
    double damage = 0.0;
    // Whether the caller has given this car the full vehicle model. Stated by the caller.
    bool fullModel = false;
    bool sighted = false;
    // Where the director wants it and how fast — what the population was told this tick, kept here
    // for a caller driving the full model. The aim is what the wheel steers at; the station is the
    // place in the player's frame the unit is holding, which for a station role is behind the aim
    // by the look-ahead. A chasing unit's two are the same point.
    glm::dvec3 aimMetres{0.0};
    glm::dvec3 stationMetres{0.0};
    double wantedSpeedMetresPerSecond = 0.0;
    // To the player, metres, this tick.
    double distanceMetres = 0.0;
    double joinedSeconds = 0.0;
    // Backing up, and why: the station is behind and close, or the unit is stuck. Both drivers read
    // `reversing`; the clocks are the director's.
    bool reversing = false;
    bool reversingToStation = false;
    double stuckSeconds = 0.0;
    double reverseSeconds = 0.0;

    // --- navigation (docs/pursuit-navigation-brief.md) ---------------------------------------------
    //
    // The line of drive to the player was clear on the last sight tick, and for how long it has been.
    bool lineClear = true;
    double lineClearSeconds = 0.0;
    // Driving the route below rather than at the player or a station.
    bool routing = false;
    PursuitRoute route{};
    SpeedProfile plan{};
    // The unit's place on the route: the segment (a hint for the next projection) and the metres.
    std::size_t routeSegment = 0;
    double routeDistanceMetres = 0.0;
    double routeAgeSeconds = 1.0e9;
    // The route's curvature one look-ahead ahead of the unit, signed left positive: the drivers'
    // feedforward. Zero for a unit that is not routing.
    double curvatureAhead = 0.0;

    // --- the radio and the search (docs/pursuit-radio-brief.md) ----------------------------------
    //
    // Near: sees the player or is inside `radioNearMetres` of it, and routes to the player itself.
    // Otherwise the unit routes to the broadcast, or, while the chase is searching, to its own search
    // goal. `goalMetres` is where the route in hand (or asked for) goes.
    bool nearPlayer = false;
    bool searching = false;
    glm::dvec3 goalMetres{0.0};
    // The search: the bearing this unit was dealt, the ring radius it is on, and the goal on it.
    bool searchBearingDealt = false;
    double searchBearingRadians = 0.0;
    double searchRadiusMetres = 0.0;
    bool searchGoalSet = false;
    glm::dvec3 searchGoalMetres{0.0};
    // The route asked for and not yet answered: the serial the answer must carry. The unit drives
    // whatever route it has meanwhile. `routeDrifted` asks for the next one at once.
    bool routePending = false;
    bool routeDrifted = false;
    std::uint64_t routeSerial = 0;

    // --- the corridor (docs/police-driving-brief.md §2.1) -------------------------------------------
    //
    // The world in front of the unit on the last sight tick: how far the rays were cast, whether any
    // hit inside that reach and how far off the nearest was, and whether the nearest is inside the
    // route distance. `corridorHit` is the flag and not a distance against the reach: the reach is
    // recomputed every tick from the unit's speed, so a unit accelerating between sight ticks would
    // otherwise read its own stale reach as an obstacle (found by the first fixture).
    double corridorReachMetres = 0.0;
    bool corridorHit = false;
    double corridorHitMetres = 0.0;
    bool corridorBlocked = false;
    // The traffic in front on the line the unit is driving for — its lane's centre, or the line it
    // is shifting onto — as the last sight tick read it: a gap and a speed along the unit's heading.
    bool trafficAhead = false;
    double trafficGapMetres = 0.0;
    double trafficSpeedMetresPerSecond = 0.0;
    // The shift onto another line (`TrafficLine`): its name — a lane and a side, a seam when the
    // side is not zero — and how far to the left of the unit's heading it was on the last sight
    // tick. Zero and not shifting when the unit drives its own lane's centre.
    bool laneShifting = false;
    std::size_t laneShiftLane = 0;
    int laneShiftSide = 0;
    double laneShiftMetres = 0.0;
    // The tail in position on the rear quarter and steering into it (the PIT); the side it went for
    // (+1 left, −1 right, 0 none yet), held for the attempt; how long the strike has run; and the
    // cooldown before the next attempt. The side a unit passes the player on, held likewise.
    bool pitting = false;
    int pitSide = 0;
    double pitStrikeSeconds = 0.0;
    double pitCooldownSeconds = 0.0;
    int passSide = 0;

    // Driving into the player at the maximum (the ram).
    bool ramming = false;
    // The block: across the player's lane, where, and for how long so far.
    bool blocking = false;
    glm::dvec3 blockPointMetres{0.0};
    double blockSeconds = 0.0;

    // --- turning round (docs/police-driving-brief.md §10) ---------------------------------------
    //
    // The room to either side on the last sight tick (the reach when nothing is there); and the
    // three-point turn under way: which way it turns (+1 left, −1 right; 0 when not turning), its
    // leg (1 forward on lock, 2 back on the opposite lock, 3 forward again), the heading turned on
    // this leg and the heading the leg ends at, the leg's clock, and the heading the turn was last
    // read at.
    double roomLeftMetres = 0.0;
    double roomRightMetres = 0.0;
    int turnSide = 0;
    int turnPhase = 0;
    double turnHeadingRadians = 0.0;
    double turnGoalRadians = 0.0;
    double turnSeconds = 0.0;
    glm::dvec3 turnLastHeading{0.0, 0.0, 1.0};
    // What the corridor capped the wanted speed to this tick, for the log and the tests; the maximum
    // unit speed when nothing did.
    double corridorCapMetresPerSecond = 0.0;
};

// What the director is told about the player each tick.
export struct PursuitPlayer
{
    // The body's origin on the road, world.
    glm::dvec3 positionMetres{0.0};
    glm::dvec3 velocityMetresPerSecond{0.0};
    glm::dvec3 forward{0.0, 0.0, 1.0};
    double lengthMetres = 4.3;
    double widthMetres = 1.8;
    // The velocity change the bodywork took this tick, m/s, split by what it hit.
    double impactPoliceMetresPerSecond = 0.0;
    double impactTrafficMetresPerSecond = 0.0;
    double impactWorldMetresPerSecond = 0.0;
};

export struct PursuitStatus
{
    bool active = false;
    // 0 to 1, and the level it reads as, 0 to 5.
    double felony = 0.0;
    int level = 0;
    bool swarm = false;
    // The chase ended with the player stopped and surrounded.
    bool busted = false;
    // The player's car has taken all the damage it can. The caller stalls it.
    bool wrecked = false;
    double playerDamage = 0.0;

    // Whether any patrol car can see the player this tick, and for how long none has.
    bool observed = false;
    double unobservedSeconds = 0.0;
    // What the player is doing this tick, watched or not; `offence` is the worst of it.
    Offence offence = Offence::None;
    bool speeding = false;
    bool offRoad = false;
    bool wrongWay = false;
    double overspeedMetresPerSecond = 0.0;

    std::size_t units = 0;
    std::size_t sightedUnits = 0;
    std::size_t fullModels = 0;
    double nearestUnitMetres = 1.0e9;
    glm::dvec3 nearestUnitMetresPosition{0.0};
    std::size_t pursuitsStarted = 0;
    std::size_t policeWrecked = 0;

    // Navigation: how many units are on a route this tick, how many routes have been asked for and
    // answered, how many are out, and the longest a search took (on whichever thread ran it).
    std::size_t routingUnits = 0;
    std::size_t routesRequested = 0;
    std::size_t routesBuilt = 0;
    std::size_t routesPending = 0;
    double routeWorstSeconds = 0.0;

    // The radio and the search (docs/pursuit-radio-brief.md): how many units route to the player,
    // how many to the broadcast, how many to a search goal; how old the last sighting is and how
    // many broadcasts there have been; whether the chase is searching, for how long, and the ring.
    std::size_t nearUnits = 0;
    std::size_t farUnits = 0;
    std::size_t searchingUnits = 0;
    double reportAgeSeconds = 0.0;
    std::size_t broadcasts = 0;
    // The general location as last broadcast: the report dead-reckoned along its lane.
    glm::dvec3 broadcastMetres{0.0};
    bool searching = false;
    double searchSeconds = 0.0;
    double searchRadiusMetres = 0.0;
    glm::dvec3 searchCentreMetres{0.0};
};

// Pedals and steering for a full model, in the vehicle model's own sense: steering −1 to 1 with a
// positive demand a right turn, throttle and brake 0 to 1.
export struct PursuitDrive
{
    double steering = 0.0;
    double throttle = 0.0;
    double brake = 0.0;
    // Reverse gear, and the throttle is the backing-up speed's.
    bool reverse = false;
};

// A full model's pose, for `PursuitDirector::drive`. The origin is the body's, on the road.
export struct PursuitCarPose
{
    glm::dvec3 positionMetres{0.0};
    glm::dquat orientation{1.0, 0.0, 0.0, 0.0};
    glm::dvec3 velocityMetresPerSecond{0.0};
    double wheelbaseMetres = 3.0;
    // The road wheel's angle at a steering demand of one, radians.
    double lockRadians = 0.46;
    // What the tyre can be asked for, as a lateral acceleration in g. Caps the speed through an arc.
    double corneringLimitG = 0.85;
};

export class PursuitDirector
{
public:
    explicit PursuitDirector(PursuitOptions options = {});

    // One tick. Reads every patrol car's pose out of the population, casts the sight rays against
    // the world, and writes an aim and a speed into every unit.
    const PursuitStatus& update(double deltaTime, const PursuitPlayer& player, TrafficPopulation& population,
                                const PhysicsWorld& world);

    [[nodiscard]] std::span<const PursuitUnit> units() const;
    [[nodiscard]] const PursuitUnit* unit(std::uint32_t agent) const;

    // The caller has stood the full vehicle model up on this unit, or taken it back.
    void setFullModel(std::uint32_t agent, bool full);
    // What a full model's bodywork took this tick, m/s of velocity change. A body the population
    // steps reports its own through `TrafficPopulation::takeImpact`.
    void reportUnitImpact(std::uint32_t agent, double metresPerSecond);

    // The driver for a full model: pure pursuit at the unit's aim, at the unit's speed, through a
    // road wheel at the stated lock.
    [[nodiscard]] PursuitDrive drive(const PursuitUnit& unit, const PursuitCarPose& pose) const;

    // Forget everything. The population's own `seed` puts its cars back; this puts the meter back.
    void reset();

    // Put the meter at a level, one to five, and start the chase if none is on — the number keys
    // (docs/police-driving-brief.md §8). Levels one to four land in the middle of their band; five is
    // the top of the meter, which is the swarm. Nothing happens on a busted or wrecked player, and a
    // level outside one to five is ignored.
    void setLevel(int level);

    // The way to the player, or null: with no router every unit drives at its aim as it always did.
    // Borrowed; the owner keeps it alive for the director's life. `setRouter` runs the router on the
    // calling thread at one search a tick (every fixture, and a driven capture); `setRouteService`
    // hands the searches to whatever stands behind the service — the game's `PursuitRouteWorker`.
    // The later call wins.
    void setRouter(PursuitRouter* router);
    void setRouteService(RouteService* service);

    [[nodiscard]] const PursuitStatus& status() const;
    [[nodiscard]] const PursuitOptions& settings() const;

private:
    struct Sight
    {
        bool sighted = false;
    };

    // The player as last called in, where it is reckoned to be by now, and the general location
    // broadcast from that.
    struct Report
    {
        bool valid = false;
        glm::dvec3 positionMetres{0.0};
        glm::dvec3 velocityMetresPerSecond{0.0};
        glm::dvec3 forward{0.0, 0.0, 1.0};
        double ageSeconds = 0.0;
        // The lane the player was on when last seen, and its speed along it, for the reckoning.
        bool onRoad = false;
        std::size_t lane = 0;
        double distanceMetres = 0.0;
        double speedAlongMetresPerSecond = 0.0;
        glm::dvec3 predictedMetres{0.0};
        bool broadcastValid = false;
        glm::dvec3 broadcastMetres{0.0};
    };

    // The search: where it is centred, which way the player was going, how long it has run.
    struct Search
    {
        bool active = false;
        double seconds = 0.0;
        glm::dvec3 centreMetres{0.0};
        glm::dvec3 heading{0.0, 0.0, 1.0};
    };

    void watch(const PursuitPlayer& player, const TrafficPopulation& population, const PhysicsWorld& world);
    void assignRoles(const PursuitPlayer& player, const TrafficPopulation& population);
    void aimUnits(const PursuitPlayer& player, TrafficPopulation& population, double deltaTime);
    void endChase(TrafficPopulation& population);
    // The radio and the search state, once a tick, before the tactics.
    void listen(const PursuitPlayer& player, bool observed, double deltaTime, const LaneNetwork& network);
    // The answers the service has finished: each installed on its unit, with the profile planned.
    void takeAnswers(const PursuitPlayer& player);
    void dealSearchBearing(PursuitUnit& unit) const;
    [[nodiscard]] glm::dvec3 searchGoalFor(PursuitUnit& unit, const LaneNetwork& network) const;
    [[nodiscard]] double ringRadius() const;

    PursuitOptions options;
    PursuitStatus summary;
    std::vector<PursuitUnit> roster;
    // The searches: the service asked, and the inline one `setRouter` stands up over a bare router.
    RouteService* service = nullptr;
    std::optional<InlineRouteService> inlineService;
    std::uint64_t routeSerials = 0;
    std::vector<RouteAnswer> answers;
    Report report;
    Search search;
    // Indexed by agent id, one per car in the population; rebuilt when the population's size changes.
    std::vector<Sight> sight;
    double sightRemainder = 1.0e9;
    double roleRemainder = 1.0e9;
    double arrestRemainder = 0.0;
    // Impacts the caller reported for full models since the last update, by agent id.
    std::vector<double> reportedImpacts;

    // Scratch the sight rays reuse.
    std::vector<glm::dvec3> rayOrigins;
    std::vector<glm::dvec3> rayDirections;
    std::vector<SurfaceHit> rayHits;
    std::vector<std::uint32_t> rayAgents;
    std::vector<double> rayDistances;
    // The line-of-drive rays ride the same batch after the sight rays: which roster entry each is.
    std::vector<std::size_t> rayRoster;
    // ...and the corridor rays after those: which roster entry each is, and how far along the
    // unit's path the ray starts (a routing unit's rays follow its route in a chain).
    std::vector<std::size_t> rayCorridor;
    std::vector<double> rayCorridorOffset;
    // ...and the room rays after those, two a unit (left, then right): which roster entry each is.
    std::vector<std::size_t> rayRoom;

    // The corridor's traffic half, on the sight tick: the lane under the unit and the shift decision.
    void readTraffic(PursuitUnit& unit, const TrafficAgent& agent, const TrafficPopulation& population);
    [[nodiscard]] double brakingFor(const PursuitUnit& unit) const;
};

} // namespace raceengine
