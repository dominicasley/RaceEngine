#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

import raceengine;

using Catch::Approx;
using raceengine::applyEnds;
using raceengine::AxisBounds;
using raceengine::AxisCalibration;
using raceengine::axisIndex;
using raceengine::axisTravel;
using raceengine::BrakeCurve;
using raceengine::brakePressure;
using raceengine::CapabilityRequest;
using raceengine::capabilityShortfall;
using raceengine::DeviceCapabilities;
using raceengine::DeviceCapability;
using raceengine::DeviceDescription;
using raceengine::deviceDriverInput;
using raceengine::DeviceIdentity;
using raceengine::DeviceProfile;
using raceengine::deviceProfileFileName;
using raceengine::deviceProfileToText;
using raceengine::DeviceSample;
using raceengine::directInputAxisRole;
using raceengine::directInputMagnitude;
using raceengine::DriverAction;
using raceengine::DriverInput;
using raceengine::EvdevAxisLayout;
using raceengine::evdevAxisLayout;
using raceengine::EvdevAxisPresence;
using raceengine::evdevAxisRole;
using raceengine::evdevGamepadActions;
using raceengine::evdevSourceKind;
using raceengine::identityFromJoystickGuid;
using raceengine::identityFromProductGuid;
using raceengine::InputAxis;
using raceengine::inputAxisFromName;
using raceengine::InputSourceKind;
using raceengine::KeyboardDemand;
using raceengine::keyboardDriverInput;
using raceengine::normaliseBipolar;
using raceengine::normaliseUnipolar;
using raceengine::parseDeviceProfile;
using raceengine::pedalForce;
using raceengine::rackFromRim;
using raceengine::seedDeviceProfile;
using raceengine::SteeringGeometry;

namespace
{

// The ClubSport V2.5 as it is on this desk, read off the device: steering 0..65535 resting at
// 32273, three pedals on the base at 0..65535 all resting at 65535, and the rotation range sysfs
// reads 900. Everything below is stated against these because a calibration that is right in the
// abstract and wrong on the hardware is not right.
constexpr auto rimMinimum = 0.0;
constexpr auto rimMaximum = 65535.0;
constexpr auto rimCentre = 32273.0;

[[nodiscard]] DeviceProfile measuredWheel()
{
    auto profile = DeviceProfile{};
    profile.identity = DeviceIdentity{.vendor = 0x0eb7, .product = 0x0004};
    profile.name = "Fanatec FANATEC ClubSport Wheel Base V2.5";
    profile.rotationDegrees = 900.0;
    profile.peakTorque = 8.0;

    profile.axes[axisIndex(InputAxis::Steering)] =
        AxisCalibration{.minimum = rimMinimum, .maximum = rimMaximum, .centre = rimCentre};

    // Resting at 65535, so that end is released and zero is full travel. Stated as an inverted pair
    // rather than as a flag beside a normal one.
    for (const auto axis : {InputAxis::Throttle, InputAxis::Brake, InputAxis::Clutch})
    {
        profile.axes[axisIndex(axis)] = AxisCalibration{.minimum = 65535.0, .maximum = 0.0, .centre = 65535.0};
    }

    profile.buttons[static_cast<std::size_t>(DriverAction::Upshift)] = 0;
    profile.buttons[static_cast<std::size_t>(DriverAction::Downshift)] = 1;
    profile.buttons[static_cast<std::size_t>(DriverAction::Handbrake)] = 2;

    return profile;
}

} // namespace

TEST_CASE("an axis maps its own ends onto zero and one", "[input][calibration]")
{
    const auto forward = AxisCalibration{.minimum = 0.0, .maximum = 65535.0};

    REQUIRE(axisTravel(forward, 0.0) == Approx(0.0));
    REQUIRE(axisTravel(forward, 65535.0) == Approx(1.0));
    REQUIRE(axisTravel(forward, 32767.5) == Approx(0.5));

    // Past either end is the end. A device that overshoots its declared range must not produce a
    // demand above one, because everything downstream treats one as full.
    REQUIRE(axisTravel(forward, -500.0) == Approx(0.0));
    REQUIRE(axisTravel(forward, 70000.0) == Approx(1.0));
}

TEST_CASE("an inverted axis is stated by which end is which and needs no flag", "[input][calibration]")
{
    // Exactly what this base's pedals do: rest at the top of the range, travel towards zero.
    const auto pedal = AxisCalibration{.minimum = 65535.0, .maximum = 0.0};

    REQUIRE(normaliseUnipolar(pedal, 65535.0) == Approx(0.0));
    REQUIRE(normaliseUnipolar(pedal, 0.0) == Approx(1.0));
    REQUIRE(normaliseUnipolar(pedal, 32767.5) == Approx(0.5));
}

TEST_CASE("a deadzone and a saturation still reach both ends", "[input][calibration]")
{
    const auto trimmed = AxisCalibration{.minimum = 0.0, .maximum = 1000.0, .deadzone = 0.1, .saturation = 0.2};

    REQUIRE(normaliseUnipolar(trimmed, 100.0) == Approx(0.0));
    // The far end is reached early and stays there rather than being scaled to 0.8.
    REQUIRE(normaliseUnipolar(trimmed, 800.0) == Approx(1.0));
    REQUIRE(normaliseUnipolar(trimmed, 1000.0) == Approx(1.0));
    REQUIRE(normaliseUnipolar(trimmed, 450.0) == Approx((0.45 - 0.1) / 0.7));

    // A trim that leaves nothing is a configuration and not a crash.
    REQUIRE(applyEnds(0.5, 0.6, 0.6) == Approx(0.0));
}

TEST_CASE("a rim off centre is scaled against each of its own halves", "[input][calibration]")
{
    const auto rim = AxisCalibration{.minimum = rimMinimum, .maximum = rimMaximum, .centre = rimCentre};

    REQUIRE(normaliseBipolar(rim, rimCentre) == Approx(0.0));
    // Both stops are full lock even though they are not the same number of counts away, which is the
    // whole reason the centre is measured rather than taken as the midpoint.
    REQUIRE(normaliseBipolar(rim, rimMaximum) == Approx(1.0));
    REQUIRE(normaliseBipolar(rim, rimMinimum) == Approx(-1.0));

    // The 494 counts between the resting position and the midpoint are worth 1.5% of lock on the
    // side they are taken from, and a rack fed the midpoint would carry that for ever.
    REQUIRE(normaliseBipolar(rim, 32767.5) == Approx(0.01488).margin(0.0005));
}

TEST_CASE("the brake reads force and not travel", "[input][pedal]")
{
    const auto cell = BrakeCurve{.sensorFullScale = 900.0, .maximumForce = 450.0, .shape = 1.0};

    REQUIRE(pedalForce(cell, 0.0) == Approx(0.0));
    REQUIRE(pedalForce(cell, 1.0) == Approx(900.0));

    // Half the cell's travel is 450 N, which this driver calls maximum braking — so the pedal is
    // fully on at half its range and the rest of the cell is headroom, not lost pressure.
    REQUIRE(brakePressure(cell, 0.5) == Approx(1.0));
    REQUIRE(brakePressure(cell, 0.25) == Approx(0.5));
    REQUIRE(brakePressure(cell, 1.0) == Approx(1.0));
}

TEST_CASE("the brake curve's shape is a preference and defaults to what the hydraulics do", "[input][pedal]")
{
    // Line pressure is proportional to pedal force through the master cylinder, and a load cell
    // already reads force — so linear is the honest default and the exponent is a driver's own
    // dial. One is the value that says nothing, and it must be exactly inert.
    const auto linear = BrakeCurve{};
    REQUIRE(linear.shape == Approx(1.0));

    const auto shaped = BrakeCurve{.sensorFullScale = 900.0, .maximumForce = 900.0, .shape = 2.0};
    REQUIRE(brakePressure(shaped, 0.5) == Approx(0.25));
    REQUIRE(brakePressure(shaped, 1.0) == Approx(1.0));

    // More of the range below half pressure is exactly what a shape above one buys.
    REQUIRE(brakePressure(shaped, 0.7) <
            brakePressure(BrakeCurve{.sensorFullScale = 900.0, .maximumForce = 900.0, .shape = 1.0}, 0.7));
}

TEST_CASE("a maximum force of nothing brakes with nothing rather than dividing by it", "[input][pedal]")
{
    REQUIRE(brakePressure(BrakeCurve{.sensorFullScale = 900.0, .maximumForce = 0.0, .shape = 1.0}, 1.0) == Approx(0.0));
}

TEST_CASE("a wheel with more travel than the car is geared one to one", "[input][steering]")
{
    // This desk: 900 degrees of device against a Mk7 GTI's 2.1 turns.
    const auto geometry = SteeringGeometry{.deviceDegrees = 900.0, .vehicleDegrees = 756.0};

    REQUIRE(rackFromRim(geometry, 0.0) == Approx(0.0));
    // Full lock arrives at 756/900 of the rim's travel and the rest is past the stops, which is
    // what the car itself does.
    REQUIRE(rackFromRim(geometry, 756.0 / 900.0) == Approx(1.0));
    REQUIRE(rackFromRim(geometry, 1.0) == Approx(1.0));
    REQUIRE(rackFromRim(geometry, -1.0) == Approx(-1.0));
    REQUIRE(rackFromRim(geometry, 0.42) == Approx(0.5));
}

TEST_CASE("a wheel with less travel than the car reaches the car's lock at its own stops", "[input][steering]")
{
    // The case one-to-one gets wrong: a 360 degree wheel geared literally could never ask for more
    // than forty percent of lock, and a car that cannot be steered is worse than one geared oddly.
    const auto geometry = SteeringGeometry{.deviceDegrees = 360.0, .vehicleDegrees = 900.0};

    REQUIRE(rackFromRim(geometry, 1.0) == Approx(1.0));
    REQUIRE(rackFromRim(geometry, 0.5) == Approx(0.5));
}

TEST_CASE("an equal range is the identity and a missing one refuses to invent a ratio", "[input][steering]")
{
    REQUIRE(rackFromRim(SteeringGeometry{.deviceDegrees = 900.0, .vehicleDegrees = 900.0}, 0.37) == Approx(0.37));
    REQUIRE(rackFromRim(SteeringGeometry{.deviceDegrees = 0.0, .vehicleDegrees = 900.0}, 0.37) == Approx(0.37));
    REQUIRE(rackFromRim(SteeringGeometry{.deviceDegrees = 900.0, .vehicleDegrees = 0.0}, 1.4) == Approx(1.0));
}

TEST_CASE("a capability that is there says nothing at all", "[input][capability]")
{
    auto capabilities = DeviceCapabilities{};
    capabilities.add(DeviceCapability::ConstantForce);

    REQUIRE(!capabilityShortfall(capabilities, CapabilityRequest{.capability = DeviceCapability::ConstantForce,
                                                                 .purpose = "force feedback",
                                                                 .fallback = "the wheel stays free"}));
}

TEST_CASE("a capability that is missing degrades to a sentence, never to a refusal", "[input][capability]")
{
    // The real case: this base's tuning menu is one bit in the driver's device table that product
    // 0x0004 does not have, so no node is created and no permission changes it.
    auto capabilities = DeviceCapabilities{};
    capabilities.add(DeviceCapability::ConstantForce);
    capabilities.add(DeviceCapability::ReadRotationRange);

    const auto notice =
        capabilityShortfall(capabilities, CapabilityRequest{.capability = DeviceCapability::TuningMenu,
                                                            .purpose = "following the base's own tuning profile",
                                                            .fallback = "the game's own settings stand alone"});

    REQUIRE(notice);
    REQUIRE(notice->find("tuning menu control") != std::string::npos);
    REQUIRE(notice->find("following the base's own tuning profile") != std::string::npos);
    REQUIRE(notice->find("the game's own settings stand alone") != std::string::npos);

    // The behavioural gate greps its log for these, and hardware that is merely different is not a
    // fault of the run.
    for (const auto* forbidden : {"error", "failed", "GL_INVALID"})
    {
        REQUIRE(notice->find(forbidden) == std::string::npos);
    }
}

TEST_CASE("gain and autocentre are asymmetric and are stated rather than assumed", "[input][capability]")
{
    // Linux: the Fanatec driver installs neither handler, so the kernel discards both with no
    // error at all. Windows: DirectInput owns both wherever it owns force feedback. A caller that
    // reached for one and was told nothing would be tuning against a knob that does not turn.
    auto throughEvdev = DeviceCapabilities{};
    throughEvdev.add(DeviceCapability::ConstantForce);
    throughEvdev.add(DeviceCapability::ReadRotationRange);
    throughEvdev.add(DeviceCapability::SetRotationRange);

    auto throughDirectInput = DeviceCapabilities{};
    throughDirectInput.add(DeviceCapability::ConstantForce);
    throughDirectInput.add(DeviceCapability::ForceGain);
    throughDirectInput.add(DeviceCapability::AutoCentre);

    REQUIRE(!throughEvdev.has(DeviceCapability::ForceGain));
    REQUIRE(throughDirectInput.has(DeviceCapability::ForceGain));
    REQUIRE(!throughDirectInput.has(DeviceCapability::ReadRotationRange));
    REQUIRE(throughEvdev.has(DeviceCapability::SetRotationRange));

    // Neither platform reaches the base's own menu without the vendor's SDK.
    REQUIRE(!throughEvdev.has(DeviceCapability::TuningMenu));
    REQUIRE(!throughDirectInput.has(DeviceCapability::TuningMenu));
}

TEST_CASE("a profile is keyed on identity and never on where the platform put the device", "[input][profile]")
{
    REQUIRE(deviceProfileFileName(DeviceIdentity{.vendor = 0x0eb7, .product = 0x0004}) == "0eb7-0004.profile");
    REQUIRE(deviceProfileFileName(DeviceIdentity{.vendor = 0x046d, .product = 0xc29b}) == "046d-c29b.profile");
}

TEST_CASE("a profile written out reads back as the same numbers", "[input][profile]")
{
    const auto original = measuredWheel();
    const auto parsed = parseDeviceProfile(deviceProfileToText(original));

    REQUIRE(parsed);
    REQUIRE(parsed->identity == original.identity);
    REQUIRE(parsed->name == original.name);
    REQUIRE(parsed->rotationDegrees == Approx(original.rotationDegrees));
    REQUIRE(parsed->peakTorque == Approx(original.peakTorque));

    for (auto index = std::size_t{0}; index < raceengine::inputAxisCount; index++)
    {
        REQUIRE(parsed->axes[index].minimum == Approx(original.axes[index].minimum));
        REQUIRE(parsed->axes[index].maximum == Approx(original.axes[index].maximum));
        REQUIRE(parsed->axes[index].centre == Approx(original.axes[index].centre));
    }

    REQUIRE(parsed->buttons == original.buttons);
}

TEST_CASE("a profile that is not one is refused rather than half read", "[input][profile]")
{
    REQUIRE(!parseDeviceProfile("identity 0eb7 0004\n"));
    REQUIRE(!parseDeviceProfile("version 1\n"));
    REQUIRE(!parseDeviceProfile("version 99\nidentity 0eb7 0004\n"));
    REQUIRE(!parseDeviceProfile("version 1\nidentity 0eb7 0004\naxis rudder 0 1 0.5 0 0\n"));
    REQUIRE(!parseDeviceProfile("version 1\nidentity 0eb7 0004\naxis brake 0 1 0.5 0.6 0.6\n"));
    REQUIRE(!parseDeviceProfile("version 1\nidentity 0eb7 0004\nrotation -900\n"));
    REQUIRE(!parseDeviceProfile("version 1\nidentity 0eb7 0004\nwobble 3\n"));
    REQUIRE(!parseDeviceProfile("version 1\nidentity zzzz 0004\n"));
}

TEST_CASE("comments and blank lines are part of the grammar", "[input][profile]")
{
    const auto parsed = parseDeviceProfile("# a wheel\nversion 1\n\nidentity 0eb7 0004   # this desk's\n"
                                           "rotation 900\n");

    REQUIRE(parsed);
    REQUIRE(parsed->identity.vendor == 0x0eb7);
    REQUIRE(parsed->rotationDegrees == Approx(900.0));
}

TEST_CASE("a seeded profile takes rest as released and as centre", "[input][profile]")
{
    auto description = DeviceDescription{};
    description.identity = DeviceIdentity{.vendor = 0x0eb7, .product = 0x0004};
    description.rotationDegrees = 900.0;

    for (auto index = std::size_t{0}; index < raceengine::inputAxisCount; index++)
    {
        description.axes[index] = AxisBounds{.minimum = 0, .maximum = 65535, .present = true};
    }

    // What this base actually reads with nothing touching it.
    auto rest = DeviceSample{};
    rest.axes[axisIndex(InputAxis::Steering)] = 32273;
    rest.axes[axisIndex(InputAxis::Throttle)] = 65535;
    rest.axes[axisIndex(InputAxis::Brake)] = 65535;
    rest.axes[axisIndex(InputAxis::Clutch)] = 65535;

    const auto seeded = seedDeviceProfile(description, rest);

    REQUIRE(seeded.axes[axisIndex(InputAxis::Steering)].centre == Approx(32273.0));
    // Resting at the top means the top is released: the ends come out inverted without anybody
    // stating a polarity that is not knowable in advance.
    REQUIRE(seeded.axes[axisIndex(InputAxis::Brake)].minimum == Approx(65535.0));
    REQUIRE(seeded.axes[axisIndex(InputAxis::Brake)].maximum == Approx(0.0));
    REQUIRE(normaliseUnipolar(seeded.axes[axisIndex(InputAxis::Brake)], 65535.0) == Approx(0.0));
    REQUIRE(seeded.rotationDegrees == Approx(900.0));
}

TEST_CASE("a seeded profile takes the other polarity when the device rests at zero", "[input][profile]")
{
    auto description = DeviceDescription{};
    description.axes[axisIndex(InputAxis::Throttle)] = AxisBounds{.minimum = 0, .maximum = 255, .present = true};

    const auto seeded = seedDeviceProfile(description, DeviceSample{});

    REQUIRE(seeded.axes[axisIndex(InputAxis::Throttle)].minimum == Approx(0.0));
    REQUIRE(seeded.axes[axisIndex(InputAxis::Throttle)].maximum == Approx(255.0));
    REQUIRE(normaliseUnipolar(seeded.axes[axisIndex(InputAxis::Throttle)], 255.0) == Approx(1.0));
}

TEST_CASE("an absent axis is left alone rather than calibrated against nothing", "[input][profile]")
{
    const auto seeded = seedDeviceProfile(DeviceDescription{}, DeviceSample{});

    // Untouched defaults, so a device with no clutch produces a clutch of zero rather than of one.
    REQUIRE(normaliseUnipolar(seeded.axes[axisIndex(InputAxis::Clutch)], 0.0) == Approx(0.0));
}

TEST_CASE("a device report becomes the same struct a keyboard produces", "[input][mapping]")
{
    const auto profile = measuredWheel();
    const auto geometry = SteeringGeometry{.deviceDegrees = 900.0, .vehicleDegrees = 756.0};

    auto sample = DeviceSample{};
    sample.axes[axisIndex(InputAxis::Steering)] = 32273;
    sample.axes[axisIndex(InputAxis::Throttle)] = 65535;
    sample.axes[axisIndex(InputAxis::Brake)] = 65535;
    sample.axes[axisIndex(InputAxis::Clutch)] = 65535;

    const auto resting = deviceDriverInput(profile, geometry, sample);

    REQUIRE(resting.steering == Approx(0.0));
    REQUIRE(resting.throttle == Approx(0.0));
    REQUIRE(resting.brake == Approx(0.0));
    REQUIRE(!resting.handbrake);

    // Rim to the right stop, throttle wide open, brake at half the cell's travel — which this
    // driver's 450 N maximum makes full pressure.
    sample.axes[axisIndex(InputAxis::Steering)] = 65535;
    sample.axes[axisIndex(InputAxis::Throttle)] = 0;
    sample.axes[axisIndex(InputAxis::Brake)] = 32767;
    sample.buttons = 0b101;

    const auto working = deviceDriverInput(profile, geometry, sample);

    REQUIRE(working.steering == Approx(1.0));
    REQUIRE(working.throttle == Approx(1.0));
    REQUIRE(working.brake == Approx(1.0));
    REQUIRE(working.upshift);
    REQUIRE(!working.downshift);
    REQUIRE(working.handbrake);
}

TEST_CASE("a button this device does not carry is never pressed", "[input][mapping]")
{
    auto profile = measuredWheel();
    profile.buttons[static_cast<std::size_t>(DriverAction::Handbrake)] = -1;

    auto sample = DeviceSample{};
    sample.buttons = ~std::uint64_t{0};

    REQUIRE(!deviceDriverInput(profile, SteeringGeometry{}, sample).handbrake);
}

TEST_CASE("both steering keys down is neither", "[input][mapping]")
{
    REQUIRE(keyboardDriverInput(KeyboardDemand{.left = true, .right = true}).steering == Approx(0.0));
    REQUIRE(keyboardDriverInput(KeyboardDemand{.left = true}).steering == Approx(-1.0));
    REQUIRE(keyboardDriverInput(KeyboardDemand{.right = true}).steering == Approx(1.0));
}

TEST_CASE("a run with nothing held produces exactly the struct a gate expects", "[input][mapping]")
{
    // Every gate this repository has runs unattended, where every key reports up. The demand that
    // comes out has to be all zeroes, or a capture would move the day input was added.
    const auto nothing = keyboardDriverInput(KeyboardDemand{});

    REQUIRE(nothing.steering == Approx(0.0));
    REQUIRE(nothing.throttle == Approx(0.0));
    REQUIRE(nothing.brake == Approx(0.0));
    REQUIRE(nothing.clutch == Approx(0.0));
    REQUIRE(!nothing.handbrake);
    REQUIRE(!nothing.upshift);
    REQUIRE(!nothing.downshift);
}

TEST_CASE("the demand is a flat struct a memcpy can move", "[input][mapping]")
{
    // Not decoration: this is the packet a rollback netcode transmits, and the day it is serialised
    // it is a memcpy or it is a schema.
    const auto original = keyboardDriverInput(KeyboardDemand{.right = true, .accelerate = true, .upshift = true});

    auto bytes = std::array<unsigned char, sizeof(DriverInput)>{};
    std::memcpy(bytes.data(), &original, sizeof(DriverInput));

    auto restored = DriverInput{};
    std::memcpy(&restored, bytes.data(), sizeof(DriverInput));

    REQUIRE(restored.steering == Approx(original.steering));
    REQUIRE(restored.throttle == Approx(original.throttle));
    REQUIRE(restored.upshift == original.upshift);
}

TEST_CASE("a joystick GUID decodes to the identity a profile is keyed on", "[input][identity]")
{
    // This wheel's own, in the layout SDL defines and the window library reports: bus, vendor,
    // product and version, each little endian and each followed by a zeroed pair. It is what a
    // joystick layer inside the window library would have to filter against to leave this device
    // to one reader.
    const auto identity = identityFromJoystickGuid("03000000b70e00000400000011010000");

    REQUIRE(identity);
    REQUIRE(identity->vendor == 0x0eb7);
    REQUIRE(identity->product == 0x0004);

    REQUIRE(!identityFromJoystickGuid("0300"));
    REQUIRE(!identityFromJoystickGuid("03000000zzzz00000400000011010000"));
}

TEST_CASE("a DirectInput product GUID decodes to the same identity", "[input][identity]")
{
    // The same wheel as Windows names it: vendor and product packed into the first field, PIDVID in
    // the last eight bytes. One profile, both platforms, no node paths anywhere.
    const auto guid = std::array<std::uint8_t, 16>{0xb7, 0x0e, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
                                                   0x00, 0x00, 'P',  'I',  'D',  'V',  'I',  'D'};
    const auto identity = identityFromProductGuid(guid);

    REQUIRE(identity);
    REQUIRE(identity->vendor == 0x0eb7);
    REQUIRE(identity->product == 0x0004);

    // Anything that is not a HID device carries a different signature and is not decoded as one.
    auto other = guid;
    other[10] = 'X';
    REQUIRE(!identityFromProductGuid(other));
}

TEST_CASE("the two platforms scale a torque differently and both are stated", "[input][capability]")
{
    // ±10000 here against evdev's ±32767. A force tuned against one and applied through the other
    // is out by more than three to one, which is the kind of difference nobody traces to a
    // constant.
    REQUIRE(directInputMagnitude(1.0) == 10000);
    REQUIRE(directInputMagnitude(-1.0) == -10000);
    REQUIRE(directInputMagnitude(0.0) == 0);
    // Rounded, not truncated: a truncation is a dead band at centre.
    REQUIRE(directInputMagnitude(0.00006) == 1);
    REQUIRE(directInputMagnitude(-0.00006) == -1);
    REQUIRE(directInputMagnitude(4.0) == 10000);
}

TEST_CASE("DirectInput's axis offsets carry the same roles evdev's codes do", "[input][identity]")
{
    REQUIRE(directInputAxisRole(raceengine::directInputOffsetX) == InputAxis::Steering);
    REQUIRE(directInputAxisRole(raceengine::directInputOffsetY) == InputAxis::Throttle);
    REQUIRE(directInputAxisRole(raceengine::directInputOffsetZ) == InputAxis::Brake);
    REQUIRE(directInputAxisRole(raceengine::directInputOffsetRz) == InputAxis::Clutch);
    REQUIRE(!directInputAxisRole(999));

    // The role names a profile is written with round-trip, which is what makes a calibration
    // portable between the two.
    REQUIRE(inputAxisFromName("brake") == InputAxis::Brake);
    REQUIRE(!inputAxisFromName("rudder"));
}

TEST_CASE("a node's layout is chosen from the whole axis set and not one code at a time", "[input][evdev]")
{
    // A wheel base: steering, three pedals, no second stick and no axis that names itself.
    REQUIRE(evdevAxisLayout(EvdevAxisPresence{}) == EvdevAxisLayout::Wheel);
    // A wired Xbox controller through `xpad`: two sticks and two triggers on Z and RZ.
    REQUIRE(evdevAxisLayout(EvdevAxisPresence{.rightStick = true}) == EvdevAxisLayout::Gamepad);
    // The same pad over Bluetooth through `hid-generic`, which names its triggers and puts the right
    // stick on Z and RZ. A device that says what a pedal is, is believed whatever else it carries.
    REQUIRE(evdevAxisLayout(EvdevAxisPresence{.namedPedals = true}) == EvdevAxisLayout::Named);
    REQUIRE(evdevAxisLayout(EvdevAxisPresence{.namedPedals = true, .rightStick = true}) == EvdevAxisLayout::Named);
}

TEST_CASE("a wheel keeps the wheel layout whatever spare axes it advertises", "[input][evdev]")
{
    // **The regression this exists for**: a base advertising ABS_RX and ABS_RY for whatever is
    // plugged into it — a shifter, a handbrake, clutch paddles — read as a pad, and a pad's Z and RZ
    // carry the two pedals the other way round. On the rig that is a throttle that brakes and a
    // brake that accelerates, with the steering, the calibration and the force feedback all normal,
    // because the *kind* was being decided next door by the two facts this now reads first.
    REQUIRE(evdevAxisLayout(EvdevAxisPresence{.rightStick = true, .statesRotation = true}) ==
            EvdevAxisLayout::Wheel);
    REQUIRE(evdevAxisLayout(EvdevAxisPresence{.rightStick = true, .takesConstantForce = true}) ==
            EvdevAxisLayout::Wheel);

    // And a base that names its pedals is still a wheel, which is the other way the axis bitmap
    // alone could have taken one: read as `Named`, its three real pedals on Y, Z and RZ drive
    // nothing and two axes nothing is standing on become the throttle and the brake.
    REQUIRE(evdevAxisLayout(EvdevAxisPresence{.namedPedals = true, .statesRotation = true}) ==
            EvdevAxisLayout::Wheel);

    // A pad states neither, so nothing above it moves: it has no lock to report and takes a rumble
    // rather than a signed torque.
    REQUIRE(evdevAxisLayout(EvdevAxisPresence{.rightStick = true}) == EvdevAxisLayout::Gamepad);
}

TEST_CASE("one evdev code carries three different roles and the layout is what says which", "[input][evdev]")
{
    // Steering is X on all three, which is the one thing a rim and two kinds of stick agree about.
    REQUIRE(evdevAxisRole(EvdevAxisLayout::Wheel, raceengine::evdevAxisX) == InputAxis::Steering);
    REQUIRE(evdevAxisRole(EvdevAxisLayout::Gamepad, raceengine::evdevAxisX) == InputAxis::Steering);
    REQUIRE(evdevAxisRole(EvdevAxisLayout::Named, raceengine::evdevAxisX) == InputAxis::Steering);

    // The measured ClubSport order, unchanged: Z is the throttle and RZ the brake, which is not the
    // order the names suggest and is the thing this file exists to keep from drifting.
    REQUIRE(evdevAxisRole(EvdevAxisLayout::Wheel, raceengine::evdevAxisY) == InputAxis::Clutch);
    REQUIRE(evdevAxisRole(EvdevAxisLayout::Wheel, raceengine::evdevAxisZ) == InputAxis::Throttle);
    REQUIRE(evdevAxisRole(EvdevAxisLayout::Wheel, raceengine::evdevAxisRz) == InputAxis::Brake);

    // ...and the pad's is the other way round, because a trigger's side is fixed by the hand on it.
    REQUIRE(evdevAxisRole(EvdevAxisLayout::Gamepad, raceengine::evdevAxisZ) == InputAxis::Brake);
    REQUIRE(evdevAxisRole(EvdevAxisLayout::Gamepad, raceengine::evdevAxisRz) == InputAxis::Throttle);
    // A pad's left stick vertical drives nothing, and nothing is a real answer: mapped to the clutch
    // the way a wheel's Y is, a thumb resting on the stick would ride the clutch all lap.
    REQUIRE(!evdevAxisRole(EvdevAxisLayout::Gamepad, raceengine::evdevAxisY));
    REQUIRE(!evdevAxisRole(EvdevAxisLayout::Gamepad, raceengine::evdevAxisRx));

    // Over Bluetooth the same two pedals are named, and Z and RZ are the right stick and must drive
    // nothing at all. Read as a wheel, this pad's right stick is its throttle and its brake.
    REQUIRE(evdevAxisRole(EvdevAxisLayout::Named, raceengine::evdevAxisGas) == InputAxis::Throttle);
    REQUIRE(evdevAxisRole(EvdevAxisLayout::Named, raceengine::evdevAxisBrake) == InputAxis::Brake);
    REQUIRE(!evdevAxisRole(EvdevAxisLayout::Named, raceengine::evdevAxisZ));
    REQUIRE(!evdevAxisRole(EvdevAxisLayout::Named, raceengine::evdevAxisRz));
}

TEST_CASE("a wheel is what it does rather than what it is called", "[input][evdev]")
{
    // Either wheel fact settles it whatever the axes look like.
    REQUIRE(evdevSourceKind(EvdevAxisLayout::Gamepad, true, false) == InputSourceKind::Wheel);
    REQUIRE(evdevSourceKind(EvdevAxisLayout::Named, false, true) == InputSourceKind::Wheel);
    // Failing both, the axes decide, and an unrecognised device with a steering axis is a wheel:
    // a pad handed a rim's shaping has no rate limit and no speed-sensitive range.
    REQUIRE(evdevSourceKind(EvdevAxisLayout::Gamepad, false, false) == InputSourceKind::Gamepad);
    REQUIRE(evdevSourceKind(EvdevAxisLayout::Named, false, false) == InputSourceKind::Gamepad);
    REQUIRE(evdevSourceKind(EvdevAxisLayout::Wheel, false, false) == InputSourceKind::Wheel);
}

TEST_CASE("a pad arrives with gears because the kernel names its face", "[input][evdev]")
{
    // What `xpad` reports, in the ascending order the backend packs its bitmap in: A, B, X, Y, the
    // two shoulders, back, start, the guide and the two stick clicks. The bit a profile binds is the
    // position in this list and not the code.
    const auto codes =
        std::vector<std::uint16_t>{0x130, 0x131, 0x133, 0x134, 0x136, 0x137, 0x13a, 0x13b, 0x13c, 0x13d, 0x13e};
    const auto actions = evdevGamepadActions(codes);

    REQUIRE(actions[static_cast<std::size_t>(DriverAction::Upshift)] == 5);
    REQUIRE(actions[static_cast<std::size_t>(DriverAction::Downshift)] == 4);
    REQUIRE(actions[static_cast<std::size_t>(DriverAction::Handbrake)] == 0);

    // A device missing one of them binds the rest and leaves that one alone rather than binding
    // whatever is nearest.
    REQUIRE(evdevGamepadActions({0x130})[static_cast<std::size_t>(DriverAction::Upshift)] == -1);
    REQUIRE(evdevGamepadActions({})[static_cast<std::size_t>(DriverAction::Handbrake)] == -1);
}

TEST_CASE("a seeded pad is a stick and two triggers rather than a rim and a load cell", "[input][profile]")
{
    // A wired Xbox controller as `xpad` states it: sticks over the signed 16-bit range resting at
    // centre, triggers 0..1023 resting released.
    auto description = DeviceDescription{};
    description.identity = DeviceIdentity{.vendor = 0x045e, .product = 0x02ea};
    description.kind = InputSourceKind::Gamepad;
    description.axes[axisIndex(InputAxis::Steering)] = AxisBounds{.minimum = -32768, .maximum = 32767, .present = true};
    description.axes[axisIndex(InputAxis::Throttle)] = AxisBounds{.minimum = 0, .maximum = 1023, .present = true};
    description.axes[axisIndex(InputAxis::Brake)] = AxisBounds{.minimum = 0, .maximum = 1023, .present = true};
    description.suggestedButtons[static_cast<std::size_t>(DriverAction::Upshift)] = 5;
    description.suggestedButtons[static_cast<std::size_t>(DriverAction::Downshift)] = 4;
    description.suggestedButtons[static_cast<std::size_t>(DriverAction::Handbrake)] = 0;

    const auto seeded = seedDeviceProfile(description, DeviceSample{});

    // A stick does not rest where it was left, so its centre is trimmed and a rim's is not.
    REQUIRE(seeded.axes[axisIndex(InputAxis::Steering)].deadzone == Approx(7849.0 / 32767.0));
    REQUIRE(normaliseBipolar(seeded.axes[axisIndex(InputAxis::Steering)], 0.0) == Approx(0.0));
    REQUIRE(normaliseBipolar(seeded.axes[axisIndex(InputAxis::Steering)], 32767.0) == Approx(1.0));
    // Full lock is still reachable: the deadzone is taken off and what is left is stretched back
    // over the whole range.
    REQUIRE(normaliseBipolar(seeded.axes[axisIndex(InputAxis::Steering)], -32768.0) == Approx(-1.0));

    // A trigger measures where it is, not how hard it is held. Through the load cell's own curve —
    // 900 N of sensor against a 450 N maximum — half a trigger would be full brake pressure.
    const auto halfTravel = normaliseUnipolar(seeded.axes[axisIndex(InputAxis::Brake)], 512.0);
    REQUIRE(halfTravel == Approx(0.5).margin(0.001));
    REQUIRE(brakePressure(seeded.brake, halfTravel) == Approx(0.5).margin(0.001));
    REQUIRE(brakePressure(seeded.brake, 1.0) == Approx(1.0));

    // Released is released and pulled is pulled, with nobody stating a polarity.
    REQUIRE(normaliseUnipolar(seeded.axes[axisIndex(InputAxis::Throttle)], 0.0) == Approx(0.0));
    REQUIRE(normaliseUnipolar(seeded.axes[axisIndex(InputAxis::Throttle)], 1023.0) == Approx(1.0));

    REQUIRE(seeded.buttons[static_cast<std::size_t>(DriverAction::Upshift)] == 5);
    REQUIRE(seeded.buttons[static_cast<std::size_t>(DriverAction::Handbrake)] == 0);
}

TEST_CASE("a seeded wheel keeps its untrimmed centre and its unbound buttons", "[input][profile]")
{
    // The change a pad brings must stop at the pad: a deadzone on a rim is what makes small
    // corrections feel like nothing, and a default binding on a base that reports a hundred and
    // eight unnamed buttons shifts the car when an unrelated one is pressed.
    auto description = DeviceDescription{};
    description.kind = InputSourceKind::Wheel;
    description.axes[axisIndex(InputAxis::Steering)] = AxisBounds{.minimum = 0, .maximum = 65535, .present = true};

    const auto seeded = seedDeviceProfile(description, DeviceSample{});

    REQUIRE(seeded.axes[axisIndex(InputAxis::Steering)].deadzone == Approx(0.0));
    REQUIRE(seeded.brake.sensorFullScale == Approx(900.0));
    REQUIRE(seeded.buttons[static_cast<std::size_t>(DriverAction::Upshift)] == -1);
}

TEST_CASE("a pad's stick reaches this car's lock and not a wheel's", "[input][mapping]")
{
    // A pad states no travel of its own, so the service makes the two travels equal and the ratio
    // one. Left at a wheel's 900 against a 540-degree car, full deflection would be reached at
    // three fifths of the stick and everything past it would be lock the driver cannot use.
    const auto geared = SteeringGeometry{.deviceDegrees = 900.0, .vehicleDegrees = 540.0};
    const auto stick = SteeringGeometry{.deviceDegrees = 540.0, .vehicleDegrees = 540.0};

    REQUIRE(rackFromRim(geared, 0.6) == Approx(1.0));
    REQUIRE(rackFromRim(stick, 0.6) == Approx(0.6));
    REQUIRE(rackFromRim(stick, 1.0) == Approx(1.0));
}
