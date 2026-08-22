export module raceengine.input;

export import :DriverInput;
export import :InputBackend;
export import :DeviceProfile;
export import :InputMapping;
// Force feedback, as three partitions and not one, because the boundary between them is the whole
// design. `:RackTorque` is the car and names no device; `:ForceMapping` is this device and is the
// only place a hardware compensation may live; `:ForceFeedbackService` is a thread and a syscall.
// Collapsing them is how a tyre model ends up quietly carrying a compensation for one base's peak
// torque.
export import :RackTorque;
export import :ForceMapping;
export import :PedalFeedback;
export import :PedalMotors;
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
export import :ForceFeedbackService;
