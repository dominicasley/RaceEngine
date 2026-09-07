module;

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

export module raceengine.input:EvdevContract;

import :DriverInput;
import :InputBackend;

namespace raceengine
{

// The facts about evdev that are numbers rather than syscalls, stated where a machine with no
// device plugged into it can compile them and a test can check them.
//
// The same split DirectInputContract keeps for the other platform, and it is worth more here. The
// axis order this engine reads a wheel with has been wrong twice, and both times invisibly: every
// calibration prompt watches one axis, so a role pointed at the wrong code reads as a pedal with no
// travel at all rather than as two pedals swapped. Stated in terms of `<linux/input.h>` macros
// inside an implementation partition, none of that can be checked without owning the hardware. The
// codes below are the kernel's own and are stable ABI — `input-event-codes.h` may add to them and
// may never renumber them.
export inline constexpr std::uint16_t evdevAxisX = 0x00;
export inline constexpr std::uint16_t evdevAxisY = 0x01;
export inline constexpr std::uint16_t evdevAxisZ = 0x02;
export inline constexpr std::uint16_t evdevAxisRx = 0x03;
export inline constexpr std::uint16_t evdevAxisRy = 0x04;
export inline constexpr std::uint16_t evdevAxisRz = 0x05;
export inline constexpr std::uint16_t evdevAxisGas = 0x09;
export inline constexpr std::uint16_t evdevAxisBrake = 0x0a;

// The pad face, as the kernel names it: BTN_SOUTH is A on an Xbox controller and BTN_TL and BTN_TR
// are the two shoulders. These are the *codes* a driver reports; the *bit* a profile binds is this
// device's own position in its ascending code list, which is what `evdevGamepadActions` converts.
export inline constexpr std::uint16_t evdevButtonSouth = 0x130;
export inline constexpr std::uint16_t evdevButtonTl = 0x136;
export inline constexpr std::uint16_t evdevButtonTr = 0x137;

// Which axis carries which role, and there are three answers rather than one.
export enum class EvdevAxisLayout : std::uint8_t {
    // A wheel base: steering on X, three pedals on Y, Z and RZ. **Measured on the ClubSport V2.5 by
    // pressing them**, and it is not the order the names suggest — there is no ABS_GAS or ABS_BRAKE
    // anywhere on that base, and its three pedals are its own 16-bit axes rather than a second USB
    // device. It was written the other way round twice before anybody pressed a pedal to check.
    Wheel,
    // A pad carrying its triggers on Z and RZ, which is what `xpad` reports for every wired Xbox
    // controller: left trigger on Z, right trigger on RZ. **The two pedals are the other way round
    // from the wheel's**, and that is not an inconsistency to tidy away — a trigger's side is fixed
    // by the hand that pulls it, a wheel's pedal order is fixed by whatever its report descriptor
    // says, and the two facts are simply different.
    Gamepad,
    // A device that names its own pedals, ABS_GAS and ABS_BRAKE. An Xbox controller over Bluetooth
    // is read by `hid-generic` and comes out this way, with Z and RZ carrying the *right stick*
    // instead — which is the whole reason a layout is chosen once for the node rather than a role
    // being guessed per code: read as a wheel, that pad's right stick is its throttle and its brake.
    Named
};

// What a node advertises, reduced to the facts the choice turns on.
export struct EvdevAxisPresence
{
    // ABS_GAS or ABS_BRAKE. A device that states what a pedal is, is believed.
    bool namedPedals = false;
    // ABS_RX and ABS_RY together, which is a right thumbstick.
    //
    // **It was written here as "the thing no wheel has", and that was wrong.** A wheel base carries
    // spare axes for whatever is plugged into it — a shifter, a handbrake, clutch paddles — and the
    // kernel gives them the codes that are free, which are these. So this is a *pad* fact only on a
    // device that has already failed to be a wheel.
    bool rightStick = false;
    // The base states a rotation range in sysfs, or its driver takes a signed constant force. Both
    // are wheel facts and neither is a pad fact: a pad has no lock to report and takes a rumble
    // rather than a torque.
    bool statesRotation = false;
    bool takesConstantForce = false;
};

// The one place a device's layout is decided.
//
// **The wheel evidence is read first, and that ordering is the whole of the fix made on 2026-09-06.**
// The axes alone decided this when it was written, and a wheel base advertising ABS_RX and ABS_RY
// for its spare inputs was therefore read as a pad — whose Z and RZ carry the two pedals **the other
// way round**. The symptom is a rig whose throttle and brake are swapped and whose steering,
// calibration and force feedback are all perfectly normal, because `evdevSourceKind` was asking the
// right question next door and getting the right answer. A wheel is a wheel first and a shape of
// axes second.
//
// What it cannot separate is stated rather than papered over: a pad that carries neither named
// pedals nor a right stick — an older two-stick pad putting its second stick on Z and RZ — reads as
// a wheel here, and no fact this backend can see says otherwise. That device needs its profile
// edited by hand, which is a worse answer than a table of product ids only until the table is a
// year old.
export [[nodiscard]] constexpr EvdevAxisLayout evdevAxisLayout(const EvdevAxisPresence presence)
{
    if (presence.statesRotation || presence.takesConstantForce)
    {
        return EvdevAxisLayout::Wheel;
    }

    if (presence.namedPedals)
    {
        return EvdevAxisLayout::Named;
    }

    if (presence.rightStick)
    {
        return EvdevAxisLayout::Gamepad;
    }

    return EvdevAxisLayout::Wheel;
}

// An evdev axis code to the role the game asks in, or nothing for an axis this layout does not
// drive the car with. Nothing is the ordinary answer and not a failure: a pad's left stick vertical,
// its right stick and its hat all land here, and an axis that drives nothing must drive nothing
// rather than the nearest role.
export [[nodiscard]] constexpr std::optional<InputAxis> evdevAxisRole(const EvdevAxisLayout layout,
                                                                      const std::uint16_t code)
{
    // X is the steering on all three, which is the one thing a rim, a stick and a pad's stick agree
    // about.
    if (code == evdevAxisX)
    {
        return InputAxis::Steering;
    }

    switch (layout)
    {
    case EvdevAxisLayout::Wheel:
        if (code == evdevAxisY)
        {
            return InputAxis::Clutch;
        }
        if (code == evdevAxisZ)
        {
            return InputAxis::Throttle;
        }
        if (code == evdevAxisRz)
        {
            return InputAxis::Brake;
        }
        break;

    case EvdevAxisLayout::Gamepad:
        // Left trigger brakes, right trigger accelerates. It is the convention every console
        // driving game keeps, and the pedals are the way round the driver's own hands are.
        //
        // A pad has no clutch and is deliberately given none: the third axis a pad could spare is
        // the left stick's vertical, and a clutch that rides on the steering hand's own stick is
        // worse than no clutch at all. The car this engine drives is a DSG.
        if (code == evdevAxisZ)
        {
            return InputAxis::Brake;
        }
        if (code == evdevAxisRz)
        {
            return InputAxis::Throttle;
        }
        break;

    case EvdevAxisLayout::Named:
        if (code == evdevAxisGas)
        {
            return InputAxis::Throttle;
        }
        if (code == evdevAxisBrake)
        {
            return InputAxis::Brake;
        }
        break;
    }

    return std::nullopt;
}

// A wheel or a pad, decided from what the device *does* rather than from a table of product ids.
//
// Rotation range and constant force are both wheel facts — a pad has no lock to state and takes no
// signed torque, only a rumble — and either one of them settles it whatever the axes look like,
// which is what keeps a wheel that states its pedals as ABS_GAS and ABS_BRAKE from being read as a
// pad. Failing both, the axes decide.
//
// It matters beyond a log line: the game shapes a demand by this answer, and a stick handed a rim's
// settings has no rate limit and no speed-sensitive range — full lock at motorway speed for a thumb
// that moved a centimetre.
export [[nodiscard]] constexpr InputSourceKind evdevSourceKind(const EvdevAxisLayout layout, const bool statesRotation,
                                                               const bool takesConstantForce)
{
    if (statesRotation || takesConstantForce)
    {
        return InputSourceKind::Wheel;
    }

    return layout == EvdevAxisLayout::Wheel ? InputSourceKind::Wheel : InputSourceKind::Gamepad;
}

// The three actions a pad can be bound without asking anybody, as bit indices into
// `DeviceSample::buttons`, and the only place in this engine that binds a button by default.
//
// A wheel's buttons are deliberately left unbound: this base reports a hundred and eight of them and
// names none, so a default would shift the car when some unrelated button was pressed, which is
// harder to diagnose than a car that will not shift. A pad is the opposite case — the kernel gives
// the face buttons their meaning — and an Xbox controller that arrives with no gears and no
// handbrake is a controller that does not work.
//
// The bit is the position in the device's own ascending code list, because that is what the backend
// packs the bitmap in and what a profile therefore names. The list is short and sorted, so this is
// a scan rather than a search: a lower_bound here would cost `<algorithm>` in the global module
// fragment of a unit that imports this module, and that is measured in seconds of build.
export [[nodiscard]] inline std::array<std::int32_t, driverActionCount>
evdevGamepadActions(const std::vector<std::uint16_t>& buttonCodes)
{
    const auto bitOf = [&buttonCodes](const std::uint16_t code)
    {
        for (auto index = std::size_t{0}; index < buttonCodes.size() && index < 64; index++)
        {
            if (buttonCodes[index] == code)
            {
                return static_cast<std::int32_t>(index);
            }
        }

        return std::int32_t{-1};
    };

    auto actions = std::array<std::int32_t, driverActionCount>{{-1, -1, -1}};
    actions[static_cast<std::size_t>(DriverAction::Upshift)] = bitOf(evdevButtonTr);
    actions[static_cast<std::size_t>(DriverAction::Downshift)] = bitOf(evdevButtonTl);
    actions[static_cast<std::size_t>(DriverAction::Handbrake)] = bitOf(evdevButtonSouth);

    return actions;
}

} // namespace raceengine
