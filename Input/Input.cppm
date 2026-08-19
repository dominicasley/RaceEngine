export module raceengine.input;

export import :DriverInput;
export import :InputBackend;
export import :DeviceProfile;
export import :InputMapping;
// The DirectInput ABI's numbers, but not DirectInput: this partition names no Windows type and
// carries no Windows header, which is what lets the Windows backend's conversions be compiled and
// tested on a machine that will never run them.
export import :DirectInputContract;
// The concrete backends are deliberately absent: :EvdevInputBackend and :DirectInputBackend are
// implementation partitions, reachable only through createInputBackend below. Exporting either
// would put <linux/input.h> or Windows.h, COM and <dinput.h> into every importer's closure — the
// same leak the Vulkan backend is kept out of raceengine.graphics to avoid.
export import :InputBackendFactory;
export import :InputService;
