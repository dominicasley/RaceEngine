export module raceengine.audio;

// Stage one and the bank, and the split between them is the same one the force feedback keeps:
// `:CarAudio` is the car in the units a bank names and knows of no device; `:SoundBank` is a car
// folder's own GUID map, which is what makes an Assetto Corsa bank addressable at all.
export import :CarAudio;
// The crossfade, which is what makes a handful of recordings sound like an engine. Pure and separate
// from every backend, because it is the part worth being sure of and being sure of it needs no device.
export import :EngineLayers;
// The tyres over it: rolling driven by the tread's own speed, skid by slip and the load under it.
// Pure for the same reason the engine mix is.
export import :TyreLayers;
// The other cars: a fleet bank's classifier, the gearbox a traffic car does not have, and the
// allocator that decides which handful of four hundred cars are audible. Pure, like the mixers.
export import :TrafficAudio;
export import :SoundBank;

// The concrete backends are deliberately absent. `:FmodAudioBackend` carries <fmod_studio.hpp> and
// `:SilentAudioBackend` is its stand-in; both are reachable only through the factories the seam
// declares, for the reason the Vulkan and evdev backends are — exporting either would put a
// third-party platform header into every importer's closure.
export import :AudioBackend;
export import :AudioService;
