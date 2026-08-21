module;

#include <cstdint>
#include <type_traits>

export module raceengine.input:DriverInput;

namespace raceengine
{

// What the driver is asking for on this tick, whatever they are holding.
//
// One struct for a rim, a stick and a pair of keys, and the physics never learns which it was. That
// is not tidiness: `VehicleInput` is what a rollback netcode transmits, and a demand shaped by the
// device it came from would make a wheel's replay and a keyboard's replay two different recordings
// of the same lap. It is trivially copyable for the same reason — the day this is serialised, it is
// a memcpy.
//
// Every axis here is the *demand* and not the thing it drives: `steering` is where the rim is, not
// the angle of the rack, and `brake` is a pressure fraction rather than the pedal's travel. The
// stage that turns a demand into a rack angle is the game's `SteeringController`, and it exists so
// that no device can assign an axis straight to the physics.
export struct DriverInput
{
    // [-1, 1], left negative. Already through the device's rotation range against the vehicle's, so
    // 1 means the car's full lock and not the rim's end stop.
    double steering = 0.0;
    // [0, 1] each.
    double throttle = 0.0;
    // Pressure, not travel. A load cell reads force at the pedal face and brake line pressure is
    // proportional to that force, so this is what the pedal is physically saying — see
    // `BrakeCurve`, where the conversion and the one number a driver actually tunes both live.
    double brake = 0.0;
    double clutch = 0.0;

    bool handbrake = false;
    // Levels rather than edges. GLFW hands the window an edge and the window reports a level, so an
    // edge recovered here would be recovered from a level anyway; the consumer that cares about
    // gear changes already keeps the previous tick's state.
    bool upshift = false;
    bool downshift = false;
};

static_assert(std::is_trivially_copyable_v<DriverInput>);

export enum class InputSourceKind : std::uint8_t {
    // Nothing is driving. What an unattended run reports, and what a game with no window and no
    // device gets.
    None,
    Keyboard,
    Gamepad,
    Wheel
};

export [[nodiscard]] constexpr const char* inputSourceKindName(const InputSourceKind kind)
{
    switch (kind)
    {
    case InputSourceKind::Keyboard:
        return "keyboard";
    case InputSourceKind::Gamepad:
        return "gamepad";
    case InputSourceKind::Wheel:
        return "wheel";
    case InputSourceKind::None:
        break;
    }

    return "nothing";
}

// One source of a `DriverInput`. Implementations differ in where the numbers come from and in
// nothing else — there is no branch anywhere downstream on which one answered.
export class IInputSource
{
public:
    IInputSource() = default;
    IInputSource(const IInputSource&) = delete;
    IInputSource(IInputSource&&) = delete;
    IInputSource& operator=(const IInputSource&) = delete;
    IInputSource& operator=(IInputSource&&) = delete;
    virtual ~IInputSource() = default;

    [[nodiscard]] virtual InputSourceKind kind() const = 0;
    // Whether anything is behind this source right now. A wheel that was switched off says no on
    // the tick after it went, and the keyboard takes over on that tick — which is why this is asked
    // every tick rather than once at startup.
    [[nodiscard]] virtual bool available() const = 0;
    // Non-const because a source may cache: `sample` is called exactly once per physics tick, and a
    // source that reads a device across a thread boundary keeps the last answer it managed to take.
    [[nodiscard]] virtual DriverInput sample() = 0;
};

} // namespace raceengine
