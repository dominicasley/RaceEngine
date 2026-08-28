export module raceengine.physics;

// A module and not a partition, and the direction of this import is the whole point: the assist
// layer cannot see vehicle state because importing this module from inside it would be a cycle. Full
// account at the head of `Assists/Api/WheelSensors.cppm`. Re-exported so that a game or a fixture
// naming a car gets the electronics fitted to it in the same import.
export import raceengine.assists;

export import :Ambient;
export import :RigidBody;
export import :JoltRuntime;
export import :ProvingGround;
export import :PhysicsWorld;
export import :Suspension;
export import :ContactPatch;
export import :Tyre;
export import :Contact;
export import :Telemetry;
export import :Brakes;
export import :Vehicle;
export import :Coupling;
export import :Clutch;
export import :Driveline;
export import :PublishedCars;
export import :SetupFile;
