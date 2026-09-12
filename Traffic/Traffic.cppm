export module raceengine.traffic;

export import :Archetypes;
export import :Dynamics;
export import :Lanes;
export import :Navmesh;
export import :Population;
export import :Pursuit;
export import :PursuitRoute;
export import :PursuitPlan;

// City traffic: a lane network, a population of drivers on it, and the rigid body one of them
// becomes when it is hit.
//
// **Its own module rather than a partition of `raceengine.physics`, and the direction of the
// dependency is the point.** This module imports physics; physics does not import this. So the
// vehicle model cannot reach traffic — it takes a span of `DynamicObstacle` and never learns that a
// traffic system exists — while traffic can reach everything it needs to put a car on the road. It
// is the shape `raceengine.assists` already keeps, for the same reason: a boundary the build
// enforces is a boundary, and one that is merely agreed is a comment.
//
// **The asset is not here either.** What this module takes is `LaneSource` — points, a limit, two
// switches — so every derivation in it can be unit tested against a network built in a fixture with
// nothing on disk. Reading Assetto Corsa's CSP traffic export and turning it into those is the
// sandbox's job, in `osr.game:TrafficNetwork`. **The one exception is the navigation mesh**
// (`:Navmesh`): five million numbers in 33 MB, read straight into structs of arrays by a scanner
// that lives beside the arrays it fills, because a hand-off through rows would be the per-polygon
// copy the format exists to avoid. It is still pure over a string, and a fixture builds one in code.
//
// The three files it is made of, and what to read first:
//
//  - `:Lanes` — the road. Lane geometry, and the three things derived from it that the export does
//    not state: which lanes loop, which run into which, and which run alongside which. Everything
//    else here is a consumer of that derivation.
//  - `:Archetypes` — the driver. A table of behaviour parameters and the three the game ships with.
//    Nothing branches on the archetype's name; adding a fourth kind of driver is an entry in a
//    table.
//  - `:Population` — the city. Four hundred cars on the network, advanced by the Intelligent Driver
//    Model with MOBIL lane changes, and the two-tier scheme that keeps that cheap.
//  - `:Dynamics` — the car that has been hit. A rigid body on four spring rays, and an explanation
//    of why it is not `stepVehicle`.
//  - `:Navmesh` — the ground. The track's drivable area as a graph of rectangles, baked for a car,
//    for a pursuing unit to route over where the lanes run out (docs/pursuit-navigation-brief.md).
//  - `:PursuitRoute` — the way there. The lanes and the mesh as one graph, A* over both, the
//    string pull and the fillet; pure over what it is handed (docs/pursuit-navigation-brief.md).
//  - `:PursuitPlan` — the inputs along the way: the speed profile of a route, its braking points and
//    the curvature the turn-in reads; pure over a polyline and a car's limits.
//  - `:Pursuit` — the police. What a patrol car sees, what counts as an offence, the felony meter,
//    who joins a chase and how the units box the player in. It steps nothing: the bodies are the
//    population's or the caller's (docs/police-pursuit-brief.md).
