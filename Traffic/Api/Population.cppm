module;

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module raceengine.traffic:Population;

import raceengine.physics;

import :Archetypes;
import :Dynamics;
import :Lanes;

namespace raceengine
{

// A city's worth of traffic, and the two tiers it is simulated in.
//
// **Tier one is a point on a lane** — a distance, a speed and an archetype — advanced by the
// Intelligent Driver Model against whatever is in front of it. It costs about fifty floating point
// operations per car per tick, so the whole of Grand City Parkway's 35 km of lane can carry four
// hundred cars at the simulation's own 360 Hz and cost less than a tenth of one vehicle tick. There
// is no spawn radius and no activity bubble: every car in the city is simulated all the time, which
// is both cheaper than the bookkeeping a bubble needs and the only way traffic ahead of the player
// is already where it should be when they arrive.
//
// **Tier two is a rigid body**, and a car enters it when something hits it hard enough — or, since
// 2026-09-09, when it is close enough to the player to be looked at: a car inside `embodyRadiusMetres`
// is a body that *follows the plan tier one goes on making for it*, so it sits on its springs, takes
// the road's camber and bumps, and pulls out of a queue by steering rather than by sliding. See
// `:Dynamics` for what that body is and why it is not the full vehicle model.
//
// Everything here is deterministic: a seeded generator, no clock read, no global touched. That is
// not tidiness — under a frame capture the simulation is driven a whole number of ticks per frame,
// and traffic that wandered would put a different city in front of the camera on every run.

export inline constexpr std::uint32_t noAgent = 0xffffffffu;
export inline constexpr std::size_t noSlot = static_cast<std::size_t>(-1);

// What a traffic car is doing.
// Who keeps the damage ledger when a contact solver's manifold is applied: the population, from the
// solver's velocity change, or the caller, from the closing speed it read off the same manifold.
export enum class ImpactLedger : std::uint8_t
{
    FromSolve,
    FromHits
};

export enum class AgentMode : std::uint8_t
{
    // On its lane, under the car-following model. The ordinary case and the cheap one.
    Cruising,
    // On its lane and under the car-following model exactly as above — and a rigid body as well,
    // because it is near the player. The lane model plans; the body follows the plan; the pose the
    // caller reads is the body's.
    Embodied,
    // Knocked off its lane and running as a rigid body, with a driver trying to get back on.
    Disturbed,
    // Still a rigid body, but the driver has given up on rejoining and is heading for the kerb.
    PullingOver,
    // Stopped where it ended up. Still a body, still an obstacle, no longer trying anything.
    Stopped,
    // A police car chasing the player as a rigid body (the same `SimpleVehicle` a wreck is), driven
    // at an aim point and a speed the pursuit director writes every tick (docs/police-pursuit-brief.md).
    Pursuing,
    // A police car whose body is somebody else's: the caller has stood the full vehicle model up on
    // it and writes its pose back every tick (`setExternalPose`). Nothing here steps it, it holds no
    // body slot, and it is left out of the obstacle list because the caller offers it with its own
    // mass. It stays an agent so the draw pool, the occupant list and the recycle rules all still see
    // one car where there is one car.
    External
};

export struct TrafficAgent
{
    std::uint32_t id = noAgent;
    AgentMode mode = AgentMode::Cruising;

    // --- where it is on the network ------------------------------------------------------------
    std::size_t lane = 0;
    double distanceMetres = 0.0;
    double speedMetresPerSecond = 0.0;
    // What this driver wants on this lane, which is the lane's limit plus their own offset and their
    // own spread. Recomputed when they change lane, because the limit may.
    double desiredSpeedMetresPerSecond = 0.0;
    // What the driver model asked for on the last tick, m/s^2. Kept for the lane-change arithmetic
    // and for anyone reading the population to find out why a car is slowing.
    double accelerationMetresPerSecondSquared = 0.0;

    // --- a lane change in progress ---------------------------------------------------------------
    std::size_t changeTarget = 0;
    // 0 to 1 across the change; zero when not changing.
    double changeProgress = 0.0;
    // How much road the change takes, metres. Progress advances by the distance covered and never by
    // the clock, so a car that is not moving does not change lanes — and the pose is a blend between
    // the two lanes' own points, so the end of the change lands on the target lane exactly.
    double changeLengthMetres = 0.0;
    // Which of this lane's neighbour runs the change is on, and the shift onto the target at the
    // car's *current* distance — re-read from the run every tick, because it changes along a bend.
    std::size_t changeRun = 0;
    double changeShiftMetres = 0.0;
    double changeCooldownSeconds = 0.0;
    // The change is being given up: the driver saw, part way across, that the gap it was moving
    // into has closed — a car coming up fast behind on the target lane, or the leader there braking
    // — and is jerking back into its own lane. The blend runs backwards at twice its rate until the
    // car is back where it started (docs/police-driving-brief.md §7).
    bool changeAborting = false;
    // A lateral offset from the lane's centre the car is still carrying — a change abandoned at a
    // join, a body that rejoined a little off centre — eased out over the road by the pose.
    double settleLateralMetres = 0.0;
    // Which way the car points, as the pose last turned it. The heading follows the motion at a rate
    // per metre covered, so a kink in the polyline or a rejoin turns the car rather than snapping it.
    glm::dvec3 heading{0.0, 0.0, 1.0};

    // How long this driver is still holding still for — a light that has just gone green, or the car
    // in front that has just moved off.
    double holdSeconds = 0.0;

    // --- the yield (docs/police-driving-brief.md §2.3) -------------------------------------------
    //
    // A siren behind this car on its lane or a neighbour: how long the yield still holds (re-armed
    // while the siren is there, counting down after), and the sideways offset toward the kerb the
    // car is carrying — eased toward its target per metre of road, like every other sideways motion
    // in the pose, and eased back out when the yield ends. Signed to the driver's left.
    double yieldSeconds = 0.0;
    double yieldLateralMetres = 0.0;
    double yieldTargetMetres = 0.0;

    // --- who is driving --------------------------------------------------------------------------
    std::size_t profile = 0;

    // --- what it looks like ----------------------------------------------------------------------
    //
    // Indices rather than a model handle or a colour, because this module draws nothing. The caller
    // owns the fleet of body shapes and the palette; these say which of each this car is, and they
    // never change for the life of the agent.
    std::uint8_t body = 0;
    std::uint8_t colour = 0;

    // --- the pose, world, valid in every mode ----------------------------------------------------
    //
    // Derived from the lane while cruising and from the rigid body otherwise, so a caller drawing or
    // colliding traffic reads these and never asks which tier a car is in.
    glm::dvec3 positionMetres{0.0};
    glm::dquat orientation{1.0, 0.0, 0.0, 0.0};
    glm::dvec3 velocityMetresPerSecond{0.0};
    // How far this car has rolled along its own heading since it was placed, metres, forward
    // positive and never wrapped. It is what a drawn wheel's angle is read off — divided by the
    // radius of whichever asset the caller draws this car with, which this module does not know.
    // Kept as a distance rather than an angle for exactly that reason, and in double so a car that
    // has driven all session still turns smoothly once the caller wraps it.
    double rolledMetres = 0.0;

    // --- tier two ---------------------------------------------------------------------------------
    std::size_t slot = noSlot;
    double disturbedSeconds = 0.0;
    double stillSeconds = 0.0;
    // For a body following its plan: how far behind the plan it is along the road, metres, never
    // negative and never past the leash. The occupant list enters the car this much behind its plan,
    // so the car following it keeps its gap to the body and not to the plan.
    double bodyLagMetres = 0.0;

    // --- the police (docs/police-pursuit-brief.md) -----------------------------------------------
    //
    // Whether this car is a patrol car: drawn at seed time against `TrafficPopulationOptions::policeBody`
    // and never changed. A patrol car cruises exactly as any other car does until the pursuit
    // director gives it something to chase.
    bool police = false;
    // Where a pursuing body is driving to and how fast, world metres and m/s, written by the pursuit
    // director every tick and read by `drivePursuit`. Meaningless in any other mode.
    glm::dvec3 pursuitAimMetres{0.0};
    double pursuitSpeedMetresPerSecond = 0.0;
    // Backing up on the director's word: toward an aim behind the car, or away from whatever the
    // car is stuck against. The speed above is then the reversing speed.
    bool pursuitReverse = false;
    // The route's curvature one look-ahead ahead, signed left positive: the driver's turn-in, on top
    // of its pure pursuit at the aim (docs/pursuit-navigation-brief.md, stage 3). Zero off a route.
    double pursuitCurvatureAhead = 0.0;
    // Backing through a three-point turn: the side the turn is to (+1 left, −1 right), so the wheel
    // goes to the opposite lock while the body backs; zero otherwise (docs/police-driving-brief.md §10).
    int pursuitTurnSide = 0;
    // Whether the light bar and the siren are on. Published for whoever draws or sounds the car;
    // nothing in this module reads it.
    bool siren = false;
    // How much velocity change this car's bodywork has absorbed from contacts since the last time
    // somebody took it, m/s, summed over hits. What a damage model reads off a body it does not own;
    // `takeImpact` reads it and zeroes it.
    double impactMetresPerSecond = 0.0;
    // Written off: the pursuit director ended this car's chase as a wreck. It never joins another
    // one — a stopped, upright wreck would otherwise be recruited again on the very tick it was
    // written off, with its damage forgotten — until the recycle pass lays it out again as a new car.
    bool wrecked = false;
};

// An outside car the traffic has to see: the player's, and any other the game is simulating itself.
//
// It is not a `DynamicObstacle` because this is the other direction — what traffic must brake for
// rather than what traffic offers to be hit. The two are different enough that sharing a type would
// mean a struct half of whose fields are unread on each side.
export struct TrafficVehicle
{
    glm::dvec3 positionMetres{0.0};
    glm::dvec3 velocityMetresPerSecond{0.0};
    glm::dvec3 forward{0.0, 0.0, 1.0};
    double lengthMetres = 4.3;
    // A patrol car in a chase with its siren on (the caller's full vehicle model). Traffic behind it
    // in the lists pulls aside for it, and the pursuit's corridor does not count it as traffic
    // (docs/police-driving-brief.md). The player's car is never this.
    bool siren = false;
};

// --- the corridor (docs/police-driving-brief.md §2.1, §7) ---------------------------------------------
//
// What the police read off the occupant lists: the traffic in front of a point on the lane under
// it, on a set of **lines** the point could drive along — the lane's own centre, each neighbour's
// centre, and the seam between the lane and each neighbour (or a seam's width out where there is no
// neighbour). A car blocks a line when its body, at the lateral offset the lists carry for it,
// overlaps the line's width; so a car that has pulled a metre and a half toward the kerb blocks its
// lane's centre and not the seam on its other side, which is how a unit threads through yielded
// traffic. *Traffic* is any occupant that is not a siren car and not an outside vehicle — the units
// ignore each other and the car they are chasing.

export struct TrafficAhead
{
    bool found = false;
    // Bumper to bumper, metres, never negative.
    double gapMetres = 0.0;
    // The car's speed **along the query's heading**, m/s: negative for a car coming the other way on
    // a lane the query is driving the wrong way, so one approach formula serves both.
    double speedMetresPerSecond = 0.0;
};

export struct TrafficLine
{
    // The line's name, stable as the unit moves between lanes: a centre is (its lane, 0); the seam
    // between two lanes is (the lower-numbered lane, the side the other is on); a seam with no
    // neighbour beyond it is (the lane, that side).
    std::size_t lane = 0;
    int side = 0;
    // From the query point to the line, metres, positive to the left of the query's heading.
    double offsetMetres = 0.0;
    TrafficAhead ahead;
    TrafficAhead behind;
};

export struct TrafficCorridor
{
    // Whether the point is on a lane at all. Nothing below means anything when it is not.
    bool onRoad = false;
    std::size_t lane = 0;
    double distanceMetres = 0.0;
    // The lane's direction at the point, and whether the query's heading runs against it.
    glm::dvec3 direction{0.0, 0.0, 1.0};
    bool wrongWay = false;
    // The point's offset from the lane's centre, metres, positive to the left of the query's heading.
    double lateralMetres = 0.0;
    // The lane's own centre line's car ahead, for a reader that wants one number.
    TrafficAhead ahead;
    // The lines: the lane's centre first, then the neighbours' centres and the seams.
    std::size_t lineCount = 0;
    std::array<TrafficLine, 5> lines{};
};

export struct TrafficUpdate
{
    double deltaTimeSeconds = 0.0;
    // Simulated seconds since the run began, which is what the lights run on. A function of the tick
    // count under a capture, and never a wall clock.
    double simulatedSeconds = 0.0;
    // Where the player is. Read only for recycling — a car is never moved to a place somebody can
    // see — and for nothing else: every agent is simulated whatever this says.
    glm::dvec3 focusMetres{0.0};
    std::span<const TrafficVehicle> vehicles;
};

export struct TrafficPopulationOptions
{
    // How many cars per kilometre of lane. Eleven puts about 380 on Grand City Parkway's 35 km,
    // which reads as a city with traffic in it rather than a city in gridlock. It is a seat number.
    double densityPerKilometre = 11.0;
    std::size_t maximumAgents = 512;

    // How many cars may be rigid bodies at once, wrecks and near cars together. Twenty-four at about
    // two microseconds each is under two per cent of the simulation's tick budget; the cap exists so
    // a pile-up cannot take the budget rather than because twenty-five would.
    std::size_t maximumDisturbed = 24;

    // How close to the player a cruising car has to come to be given a body, and how far a body has
    // to go before it is a point again. Zero is off, which is the default and what every test runs
    // under; the game sets the first from its nearest level of detail and the second a little past
    // it, so a car does not flicker between the tiers at the boundary. Nothing happens either way
    // when the update names no outside vehicle: the focus means nothing without one.
    double embodyRadiusMetres = 0.0;
    double disembodyRadiusMetres = 0.0;

    // How hard a car has to be hit before it stops being a point on a lane, m/s of velocity change.
    // Below it a nudge is ignored — a mirror clipped in passing is not a crash, and promoting on one
    // would leave a trail of disturbed cars behind any driver who uses the whole lane.
    double impactSpeedThresholdMetresPerSecond = 0.40;

    // What it takes to be back on the lane: close enough, pointed the right way, upright, rolling —
    // and **not still crossing it**.
    //
    // The last of those is not a detail. A car that has just been shoved sideways is, for the first
    // tick, still exactly where it was: on its lane, pointing along it, with 4.5 m/s of lateral
    // velocity it has not travelled on yet. Without a test on that velocity the recovery rejoins on
    // the tick it was promoted, every time, and the whole tier is dead code that looks like it
    // works.
    double rejoinLateralMetres = 1.30;
    double rejoinHeadingAgreement = 0.94;
    double rejoinLateralSpeedMetresPerSecond = 0.60;
    // And a floor on how long a car stays a body, so a graze does not flicker between the two tiers
    // on alternate ticks.
    double minimumDisturbedSeconds = 0.50;
    // How long a driver tries to rejoin before giving up and heading for the kerb, seconds.
    double recoverySeconds = 9.0;
    // How slow a car has to be, m/s, and for how long, before it counts as stopped.
    double stillSpeedMetresPerSecond = 0.35;
    double stillSeconds = 2.5;

    // How far a stopped car has to be from the player before it may be recycled, and how long it has
    // to have been stopped...
    double recycleDistanceMetres = 180.0;
    double recycleAfterSeconds = 45.0;
    // ...and how far when it is *ahead* of them, where a straight city street shows a car at three
    // hundred metres. A car is taken from, and put down at, a point only when the point passes both:
    // behind the player past the first distance, ahead of them past the second. Without the second a
    // stopped car two hundred metres up the road vanished, and another appeared in its place.
    double recycleAheadMetres = 450.0;

    // Decisions — lane changes, the per-lane ordering, the projections — run on their own clock
    // rather than every tick. Thirty a second is finer than a driver decides at.
    double decisionIntervalSeconds = 1.0 / 30.0;

    // How far ahead a driver looks for a leader, a signal or the end of the road, metres.
    double lookAheadMetres = 140.0;

    // The car, for both tiers: tier one uses the length and width to keep gaps and to hand out
    // obstacles, and tier two uses the whole of it.
    SimpleVehicleSetup vehicle;
    // What the bodywork does when it meets something.
    ContactMaterial contact;

    // How many body shapes and how many colours the caller has. The agent's own indices are drawn
    // against these at seed time and never change.
    std::uint8_t bodyCount = 1;
    std::uint8_t colourCount = 8;

    // --- the police (docs/police-pursuit-brief.md) -----------------------------------------------
    //
    // Which of the caller's body shapes is the patrol car, or `noPoliceBody` for a fleet with none —
    // every earlier test and every track without a police car in its fleet. With one stated, a car is
    // a patrol car with probability `policeShare` and takes that body; every other car draws its body
    // uniformly from the *other* shapes, so the police are a stated fraction of the city rather than
    // a fifth of it.
    std::uint8_t policeBody = 0xff;
    double policeShare = 0.10;

    // What a pursuing body has that a traffic body does not: the drive and brake force to chase, and
    // the tyre to corner while doing it. The geometry — box, wheelbase, rays — stays `vehicle`'s, so
    // the batched ray cast and the obstacle box are the same code for every body; only the forces
    // differ. 14 kN on 1750 kg is 8 m/s², which is a V8 interceptor driven hard.
    double pursuitDriveForceNewtons = 14000.0;
    double pursuitBrakeForceNewtons = 18000.0;
    double pursuitFrictionCoefficient = 1.10;
    // A pursuing body may run past the speed a traffic car is dropped at. Ninety metres a second is
    // above anything the player's car reaches.
    double pursuitSpeedLimitMetresPerSecond = 90.0;
    // The pursuing body's driver caps its speed for a corner in its path — the turn-in the director
    // wrote, and the aim's own bearing only when it is more than this far off the nose — and
    // corrects a small error with the wheel at the curvature the tyre holds at speed, never by
    // slowing (docs/police-driving-brief.md §9). The same number as `PursuitOptions`'s.
    double pursuitSteeringCapAngleRadians = 0.35;

    // --- the yield (docs/police-driving-brief.md §2.3, §7) ----------------------------------------
    //
    // A cruising car with a siren car behind it — on its own lane or a neighbour, within the far
    // reach and closing, or within the near reach whatever it is doing — caps its desired speed at
    // this share of the lane's limit (floored), moves this far toward the kerb over this much road,
    // starts no lane change, and keeps all of that for the hold after the siren is gone. The kerb is
    // the side with no neighbour; a lane with neighbours both sides, or none, pulls to the stated
    // side (−1 is the driver's right, which is where a Charger city drives — placed, no data). The
    // pull is 1.4 m since §7: a car's width and a half off its centre is what leaves the seam on its
    // other side clear for a unit to thread through. All placed; the seat is the verifier.
    double yieldBehindMetres = 80.0;
    double yieldNearMetres = 25.0;
    double yieldSpeedFactor = 0.5;
    double yieldSpeedFloorMetresPerSecond = 4.0;
    double yieldLateralMetres = 1.4;
    double yieldLengthMetres = 12.0;
    double yieldHoldSeconds = 2.5;
    double yieldSideWithNoNeighbour = -1.0;

    // --- the occupant lists' width (docs/police-driving-brief.md §7) -------------------------------
    //
    // How far off a lane's centre an outside vehicle or a body off its lane is still looked for on
    // the network; it is then entered on that lane, and on each neighbour, whose width its body
    // overlaps — the body's span across the road read off its heading, so a patrol car turning
    // round across two lanes is in both lists and the traffic on either brakes for it. Four metres
    // and one lane, before, left a player or a patrol car on a seam in neither, and a car sideways
    // across the next lane out of that lane's list (docs/police-driving-brief.md §7, §9).
    double occupantLateralMetres = 6.0;
    double laneHalfWidthMetres = 1.85;
    // A siren car ahead of a cruising car is followed at this standstill gap rather than the
    // driver's own: the traffic stops short of a patrol car in its lane and leaves it room to turn
    // (docs/police-driving-brief.md §9, Dominic: "once police are in front of cars they should stop
    // in their lane until the police is gone").
    double sirenStandoffMetres = 8.0;
    // The lines a unit may drive: the seam is put this far out where a lane has no neighbour on that
    // side (half a lane's width), and a car and a line are clear of each other by this much.
    double seamWithoutNeighbourMetres = 1.85;
    double lineClearanceMetres = 0.35;
    // How far back the merge safety looks for a car coming up on the lane being joined, both for
    // the traffic's own lane changes and for the police's corridor: a patrol car at sixty metres a
    // second is two hundred metres away three seconds before it arrives.
    double mergeLookBackMetres = 250.0;

    // The generator's seed. Fixed, so two runs of the same capture put the same cars in the same
    // places.
    std::uint64_t seed = 0x9E3779B97F4A7C15ULL;
};

export inline constexpr std::uint8_t noPoliceBody = 0xff;

// What one update did, for a log line and for a test.
export struct TrafficReport
{
    std::size_t agents = 0;
    std::size_t cruising = 0;
    // Bodies following their plan, near the player.
    std::size_t embodied = 0;
    std::size_t disturbed = 0;
    std::size_t stopped = 0;
    // Police bodies chasing the player, and police cars whose body is the caller's full model.
    std::size_t pursuing = 0;
    std::size_t external = 0;
    std::size_t changingLanes = 0;
    std::size_t recycled = 0;
    // How many cars were promoted out of tier one on this update.
    std::size_t promoted = 0;
    // How many wanted to be and could not, because the pool was full. A non-zero number here is the
    // one thing that says `maximumDisturbed` is too small.
    std::size_t refused = 0;
    // Near cars left as points this update because a body could not be stood up where their plan is:
    // no ground under a wheel, or the world already inside the box. Cumulative over the run, not per
    // update, because it is a map diagnostic — a lane that runs through a kerb or a wall shows here.
    std::size_t heldAsPoints = 0;
    // Bodies taken off the road this update because the physics would have thrown them: a following
    // body its plan had pushed deep into geometry, or any body past a speed a traffic car cannot
    // reach. Cumulative, for the same reason.
    std::size_t dropped = 0;
};

export class TrafficPopulation
{
public:
    TrafficPopulation(LaneNetwork laneNetwork, TrafficPopulationOptions settings, std::vector<DriverProfile> drivers);

    // Fill the network. Idempotent: calling it again clears what is there and lays the same cars out
    // again, because the generator is reseeded.
    void seed();

    // One tick. `world` is queried only for the disturbed cars' wheel rays and bodywork, and only
    // when there are any — a city where nothing has been hit makes no query at all.
    const TrafficReport& update(const TrafficUpdate& input, const PhysicsWorld& world);

    // Every car, in a stable order. The pose fields are valid whatever tier a car is in.
    [[nodiscard]] std::span<const TrafficAgent> agents() const;

    // The cars within `radiusMetres` of a point, as obstacles the vehicle model can hit. Appended to
    // `into`, which the caller is expected to reuse so a tick allocates nothing.
    void obstaclesNear(const glm::dvec3& pointMetres, const double radiusMetres,
                       std::vector<DynamicObstacle>& into) const;

    // Take back what a contact solver did to those obstacles. Anything above the impact threshold
    // promotes that car out of tier one, carrying the velocity the solve gave it.
    //
    // Called once per manifold, after the vehicle that produced it has been stepped and before the
    // next update. A manifold with no obstacle bodies in it does nothing at all.
    //
    // What a hit *costs* a car — the impact `takeImpact` hands a damage model — is the solver's own
    // velocity change on the body by default, which carries the position correction on every tick
    // of an overlap (54 m/s per metre, docs/police-pursuit-brief.md §9). A caller that has read the
    // closing speed off the manifold says `ImpactLedger::FromHits`, and charges each car itself
    // through `chargeImpact` — once, at the speed the pair met. The nudge and the promotion are the
    // same either way.
    void applyContacts(const ContactManifold& manifold, ImpactLedger ledger = ImpactLedger::FromSolve);
    void chargeImpact(std::uint32_t id, double metresPerSecond);

    // --- the police (docs/police-pursuit-brief.md) -----------------------------------------------
    //
    // A patrol car sent after the player. A point on its lane is stood up as a body where it is (on
    // the same terms `embody` stands one up: ground under every wheel, nothing inside the box — and
    // refused, with the refusal counted, where that fails or the pool is full); a car that is already
    // a body keeps it. Its aim and speed are whatever the director last wrote, so a caller sets them
    // first. False when the id names no car, the car is not a patrol car, or no body could be had.
    [[nodiscard]] bool beginPursuit(std::uint32_t id, const PhysicsWorld& world);
    // Where a pursuing car is driving to, how fast it may go getting there, and whether its lights
    // are on. Read by `drivePursuit` for a body and published for an external car; ignored on any
    // car that is not pursuing.
    void setPursuitAim(std::uint32_t id, const glm::dvec3& aimMetres, double speedMetresPerSecond, bool siren,
                       bool reverse, double curvatureAhead = 0.0, int turnSide = 0);
    // The chase is over for this car. A wrecked one stops where it is; the rest are handed to the
    // recovery driver, which puts them back on a lane the way it puts back any other wreck. An
    // external car is given a body at its last pose first.
    void endPursuit(std::uint32_t id, bool wrecked);
    // Hand a pursuing car's body to the caller: its slot is freed, its pose stays, and from here the
    // caller writes the pose. The agent's pose fields are where the caller should stand its own model
    // up. False unless the car is pursuing.
    [[nodiscard]] bool takeExternal(std::uint32_t id);
    void setExternalPose(std::uint32_t id, const glm::dvec3& positionMetres, const glm::dquat& orientation,
                         const glm::dvec3& velocityMetresPerSecond, double rolledMetres);
    // The caller has finished with the car: it is a pursuing body again, stood up at the pose the
    // caller last wrote. A body that cannot be stood up there leaves the car a stopped wreck.
    void releaseExternal(std::uint32_t id, const PhysicsWorld& world);
    // The velocity change this car's bodywork has taken from contacts since the last call, m/s,
    // and zero it. What a damage model reads.
    [[nodiscard]] double takeImpact(std::uint32_t id);
    // Every patrol car's id, in pool order, for a director that has to look at each of them.
    [[nodiscard]] std::span<const std::uint32_t> policeIds() const;
    // The traffic in front of a point going a way, off the occupant lists as they stood on the last
    // decision tick (docs/police-driving-brief.md §2.1): the lane under the point in either direction
    // within `lateralMetres`, the nearest traffic car ahead on it within `reachMetres`, and on each
    // neighbour running beside it there the nearest ahead and the nearest behind. Siren cars and
    // outside vehicles are not traffic.
    [[nodiscard]] TrafficCorridor trafficAround(const glm::dvec3& pointMetres, const glm::dvec3& heading,
                                                double lateralMetres, double reachMetres) const;

    [[nodiscard]] const LaneNetwork& network() const;
    [[nodiscard]] const TrafficPopulationOptions& settings() const;
    [[nodiscard]] const std::vector<DriverProfile>& driverProfiles() const;
    [[nodiscard]] const TrafficReport& report() const;

    // Which profile each name maps to, for a log line. Empty until `seed` has run.
    [[nodiscard]] std::vector<std::size_t> profileCounts() const;

private:
    // One car on one lane as the car behind it sees it: a place, a speed and a length. Agents,
    // outside vehicles and the virtual obstacles a red light or a dead end stands in for all become
    // one of these, which is what lets the car-following model have exactly one leader lookup.
    struct Occupant
    {
        double distanceMetres = 0.0;
        double speedMetresPerSecond = 0.0;
        double lengthMetres = 0.0;
        std::uint32_t agent = noAgent;
        // A patrol car in a chase: a pursuing body, or an outside vehicle the caller marked. What the
        // yield reads, and what the pursuit's corridor leaves out.
        bool siren = false;
        // Where the car's centre stands across the lane, metres, positive to the driver's left of the
        // lane's centre: a yield, a settle, a change in progress, or a body off its lane. What lets
        // the corridor tell a car that has pulled aside from one that has not.
        double asideMetres = 0.0;
    };

    // One rigid body, whichever tier-two reason it exists for.
    struct DisturbedSlot
    {
        bool live = false;
        std::uint32_t agent = noAgent;
        SimpleVehicleState state;
    };

    [[nodiscard]] double nextRandom();
    [[nodiscard]] std::size_t pickProfile();

    void rebuildLaneOccupants(const TrafficUpdate& input);
    void insertOccupant(const std::size_t lane, const Occupant& occupant);

    // The car-following model's answer for one agent against one leader gap.
    [[nodiscard]] double followAcceleration(const DriverBehaviour& behaviour, const double speed,
                                            const double desiredSpeed, const double gapMetres,
                                            const double approachMetresPerSecond) const;

    // What is in front of an agent on a lane: the gap and the leader's speed. `found` is false when
    // the road ahead is clear within the look-ahead.
    struct Ahead
    {
        bool found = false;
        double gapMetres = 0.0;
        double speedMetresPerSecond = 0.0;
        // Whether what is in the way is a light rather than a car. It decides which of the driver's
        // two reaction delays applies when they are stopped behind it.
        bool signal = false;
        // A patrol car in a chase: followed at the siren standoff.
        bool siren = false;
    };

    [[nodiscard]] Ahead leaderOn(const std::size_t lane, const double distanceMetres, const std::uint32_t ignore) const;
    // `reachMetres` under zero is the ordinary look-ahead; the merge checks look further back.
    [[nodiscard]] Ahead followerOn(const std::size_t lane, const double distanceMetres, const std::uint32_t ignore,
                                   const double reachMetres = -1.0) const;
    // Whether the gap a changing car is moving into on its target lane is still safe: the leader
    // there not about to be hit, the follower there not forced past the safe deceleration.
    [[nodiscard]] bool changeStillSafe(const TrafficAgent& agent, const DriverBehaviour& behaviour) const;
    // One car's lateral offset from a lane's centre at a distance, off its pose.
    [[nodiscard]] double asideOf(const TrafficAgent& agent, const std::size_t lane, const double distanceMetres) const;
    [[nodiscard]] Ahead signalAhead(const std::size_t lane, const double distanceMetres, const double seconds) const;
    [[nodiscard]] Ahead laneEndAhead(const std::size_t lane, const double distanceMetres) const;
    // The nearest siren car behind a place on a lane within a reach, for the yield.
    [[nodiscard]] Ahead sirenBehindOn(const std::size_t lane, const double distanceMetres, const double reachMetres) const;
    // The yield, on the decision clock: whether a siren is behind this car on its lane or a
    // neighbour, and which way the kerb is (docs/police-driving-brief.md §2.3).
    void considerYield(TrafficAgent& agent);

    void updateCruising(TrafficAgent& agent, const TrafficUpdate& input, const bool decide);
    void considerLaneChange(TrafficAgent& agent, const double ownAcceleration);
    // The pose off the lane, given the road covered this tick: the heading turns at a rate per metre
    // and every sideways motion is stated per metre, so a car that is not moving does not turn.
    // `snapHeading` puts the heading straight on, for a car that has just been placed.
    void refreshCruisingPose(TrafficAgent& agent, const double travelMetres, const bool snapHeading);
    // Read the lateral residual between where the car is and the lane it now states, so the pose
    // eases it out rather than dropping it.
    void settleOntoLane(TrafficAgent& agent);

    // Where the plan puts the car `aheadMetres` further along its road than it is now, blend, settle
    // and all. What the body's steering aims at.
    [[nodiscard]] std::optional<glm::dvec3> plannedPosition(const TrafficAgent& agent, const double aheadMetres) const;

    // The near tier: give a cruising car inside the embody radius a body, take it back from one
    // outside the other. After the plan and before the bodies step.
    void embodyNear(const TrafficUpdate& input, const PhysicsWorld& world);
    [[nodiscard]] bool embody(TrafficAgent& agent, const PhysicsWorld& world);
    // Take a body away and leave the car a point on its plan, where its pose already is.
    void dropBody(TrafficAgent& agent);
    void disembody(TrafficAgent& agent);
    // A wreck that has found its lane again while still near the player: it keeps its body and picks
    // up a plan, where a far one goes back to being a point.
    void reembody(TrafficAgent& agent, const LaneProjection& projection, const double speed);
    [[nodiscard]] std::size_t freeBodySlot() const;

    // Every live body, one tick: the near ones following their plan, the wrecks under recovery.
    void updateBodies(const TrafficUpdate& input, const PhysicsWorld& world);
    void driveFollowingPlan(const TrafficAgent& agent, const DisturbedSlot& slot, SimpleVehicleInput& into) const;
    void driveRecovery(TrafficAgent& agent, DisturbedSlot& slot, SimpleVehicleInput& into,
                       const TrafficUpdate& input) const;
    // The police driver for a body: pure pursuit at the aim the director wrote, at the speed it
    // wrote, capped by what the tyre can corner at.
    void drivePursuit(const TrafficAgent& agent, const DisturbedSlot& slot, SimpleVehicleInput& into) const;
    // Stand a body up at the agent's own pose, on `embody`'s terms. The mode is the caller's to set.
    [[nodiscard]] bool standUpBody(TrafficAgent& agent, const PhysicsWorld& world);

    [[nodiscard]] bool promote(TrafficAgent& agent, const glm::dvec3& extraVelocity,
                               const glm::dvec3& extraAngularVelocity);
    void demote(TrafficAgent& agent, const LaneProjection& projection, const double speed);

    void recycle(TrafficAgent& agent, const glm::dvec3& focusMetres, const glm::dvec3& focusForward);
    [[nodiscard]] bool placeOnNetwork(TrafficAgent& agent, const glm::dvec3& focusMetres,
                                      const glm::dvec3& focusForward, const bool awayFromFocus);
    // Whether a point is somewhere the player cannot watch a car appear or vanish. `focusForward` may
    // be zero, which makes every direction "behind".
    [[nodiscard]] bool outOfSight(const glm::dvec3& pointMetres, const glm::dvec3& focusMetres,
                                  const glm::dvec3& focusForward) const;

    LaneNetwork lanes;
    TrafficPopulationOptions options;
    // `options.vehicle` with the pursuit forces on it — built once here so the geometry can never
    // differ between the two.
    SimpleVehicleSetup pursuitVehicle;
    std::vector<DriverProfile> profiles;
    // The patrol cars' ids, filled at seed time.
    std::vector<std::uint32_t> police;
    // The cumulative weights the profile draw reads, normalised at construction.
    std::vector<double> profileWeights;

    std::vector<TrafficAgent> pool;
    // Indexed by lane, each sorted by distance. Rebuilt on the decision clock.
    std::vector<std::vector<Occupant>> occupants;
    std::vector<DisturbedSlot> disturbed;

    // Scratch the tick reuses, so a city where nothing has been hit allocates nothing.
    std::vector<glm::dvec3> rayOrigins;
    std::vector<glm::dvec3> rayDirections;
    std::vector<SurfaceHit> rayHits;
    std::vector<DynamicObstacle> obstacleScratch;
    ContactManifold manifoldScratch;

    std::uint64_t generator = 0;
    double decisionRemainder = 0.0;
    std::uint32_t nextId = 0;
    TrafficReport summary;

    // Promotions and refusals happen inside `applyContacts`, which runs between updates rather than
    // inside one. They are tallied here and folded into the report the next update produces, so a
    // caller reading the report sees the whole tick rather than the half of it that happened after
    // the last thing it called.
    std::size_t promotedSinceReport = 0;
    std::size_t refusedSinceReport = 0;
    std::size_t heldAsPointsTotal = 0;
    std::size_t droppedTotal = 0;
};

} // namespace raceengine
