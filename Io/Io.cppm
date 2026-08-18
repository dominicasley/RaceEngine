export module raceengine.io;

// raceengine.io.accessor is deliberately absent: glTF byte-slicing is the loader's business, and
// :GLTFService imports it without re-exporting it, so it stays reachable to this module and
// invisible to importers of raceengine.io.
export import :GLTFService;
export import :ResourceService;
