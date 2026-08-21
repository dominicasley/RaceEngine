#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine;

using raceengine::AxisCalibration;
using raceengine::axisIndex;
using raceengine::deviceDriverInput;
using raceengine::DeviceForceProfile;
using raceengine::DeviceProfile;
using raceengine::DeviceSample;
using raceengine::directInputAxisMaximum;
using raceengine::directInputAxisMinimum;
using raceengine::directInputAxisRole;
using raceengine::directInputMagnitude;
using raceengine::directInputOffsetRz;
using raceengine::directInputOffsetX;
using raceengine::directInputOffsetY;
using raceengine::directInputOffsetZ;
using raceengine::DriverInput;
using raceengine::ForceMapping;
using raceengine::InputAxis;
using raceengine::inputAxisCount;
using raceengine::mapRackTorque;
using raceengine::SteeredCorner;
using raceengine::SteeringGeometry;
using raceengine::SteeringRack;
using raceengine::steeringRackTorque;

namespace
{

// The same wheel, described the way each platform describes it. Both report a full sixteen-bit
// range on this base; what differs is which name the axis arrives under, and that is exactly the
// thing a parity test has to hold still.
[[nodiscard]] DeviceProfile profileWithBounds(const std::int32_t minimum, const std::int32_t maximum)
{
    auto profile = DeviceProfile{};

    for (auto index = std::size_t{0}; index < inputAxisCount; index++)
    {
        profile.axes[index] = AxisCalibration{.minimum = static_cast<double>(minimum),
                                              .maximum = static_cast<double>(maximum),
                                              .centre = 0.5,
                                              .deadzone = 0.0,
                                              .saturation = 0.0};
    }

    return profile;
}

// A lap's worth of driver, as raw device codes. Deterministic and written here rather than captured,
// because what this replays has to be the *same* sequence on both platforms by construction.
[[nodiscard]] std::vector<DeviceSample> replay(const std::int32_t minimum, const std::int32_t maximum)
{
    const auto span = static_cast<double>(maximum - minimum);

    auto samples = std::vector<DeviceSample>{};
    samples.reserve(600);

    for (auto step = 0; step < 600; step++)
    {
        const auto phase = static_cast<double>(step) / 600.0;

        auto sample = DeviceSample{};
        // A steering sweep that reaches both locks, a throttle that comes and goes, and a brake
        // pressed hard enough to be past the load cell's curve rather than on its toe.
        sample.axes[axisIndex(InputAxis::Steering)] =
            minimum + static_cast<std::int32_t>(span * (0.5 + 0.5 * std::sin(phase * 6.2831853071795865)));
        sample.axes[axisIndex(InputAxis::Throttle)] =
            minimum + static_cast<std::int32_t>(span * (0.5 + 0.5 * std::cos(phase * 12.566370614359172)));
        sample.axes[axisIndex(InputAxis::Brake)] = minimum + static_cast<std::int32_t>(span * 0.75 * phase);
        sample.axes[axisIndex(InputAxis::Clutch)] = minimum;

        samples.push_back(sample);
    }

    return samples;
}

// Stage one for a rack sitting where the driver put it. The geometry is fixed, so what varies over
// the replay is the rack travel and nothing else — which is the point: this is the trace that has to
// match, and it is in newton metres.
[[nodiscard]] double rackTorqueAt(const double rackTravel)
{
    const auto corner = SteeredCorner{.lowerBallJoint = glm::dvec3(0.75, 0.10, 0.0),
                                      .upperBallJoint = glm::dvec3(0.69, 0.60, 0.0),
                                      .steeringArm = glm::dvec3(0.75, 0.10, -0.15),
                                      .rackOuter = glm::dvec3(0.30 + rackTravel, 0.10, -0.15),
                                      .contactPatch = glm::dvec3(0.77, 0.0, 0.04),
                                      .patchNormal = glm::dvec3(0.0, 1.0, 0.0),
                                      // A side force that follows the lock, which is what a tyre does.
                                      .tyreForce = glm::dvec3(4000.0 * rackTravel / 0.055, 4000.0, 0.0),
                                      .aligningMoment = -100.0 * rackTravel / 0.055};

    const auto corners = std::array<SteeredCorner, 1>{corner};

    return steeringRackTorque(SteeringRack{}, std::span<const SteeredCorner>(corners), 0.0).steeringTorque;
}

} // namespace

TEST_CASE("DirectInput's axis offsets name the same axes evdev's codes do", "[input][ffb][parity]")
{
    // Where the two platforms genuinely disagree, and therefore the one place a divergence could be
    // introduced without anybody noticing: a wheel whose throttle and brake are swapped on Windows
    // is a car that brakes when it is asked to go, and every trace after that point is a different
    // lap rather than a different platform.
    REQUIRE(directInputAxisRole(directInputOffsetX) == InputAxis::Steering);
    REQUIRE(directInputAxisRole(directInputOffsetY) == InputAxis::Throttle);
    REQUIRE(directInputAxisRole(directInputOffsetZ) == InputAxis::Brake);
    REQUIRE(directInputAxisRole(directInputOffsetRz) == InputAxis::Clutch);

    // And an offset that is none of them is answered as none of them rather than as the first.
    REQUIRE_FALSE(directInputAxisRole(0xdeadbeef).has_value());
}

TEST_CASE("an identical input sequence produces an identical stage-one trace on either backend", "[input][ffb][parity]")
{
    // Criterion 12. Stage one is in newton metres at the rim and knows nothing about what is
    // plugged in, so a divergence here is a bug by definition — where the platforms are *allowed* to
    // differ is stage two, and that is asserted below rather than left implied.
    const auto geometry = SteeringGeometry{};

    const auto evdevProfile = profileWithBounds(0, 65535);
    const auto windowsProfile = profileWithBounds(directInputAxisMinimum, directInputAxisMaximum);

    const auto onLinux = replay(0, 65535);
    const auto onWindows = replay(directInputAxisMinimum, directInputAxisMaximum);

    REQUIRE(onLinux.size() == onWindows.size());

    auto linuxTrace = std::vector<double>{};
    auto windowsTrace = std::vector<double>{};

    for (auto index = std::size_t{0}; index < onLinux.size(); index++)
    {
        const auto through = deviceDriverInput(evdevProfile, geometry, onLinux[index]);
        const auto across = deviceDriverInput(windowsProfile, geometry, onWindows[index]);

        // The driver's own packet first: if this differs, the trace below differs for a reason that
        // has nothing to do with the steering.
        REQUIRE(through.steering == across.steering);
        REQUIRE(through.throttle == across.throttle);
        REQUIRE(through.brake == across.brake);
        REQUIRE(through.clutch == across.clutch);

        linuxTrace.push_back(rackTorqueAt(through.steering));
        windowsTrace.push_back(rackTorqueAt(across.steering));
    }

    SECTION("bit for bit, and not approximately")
    {
        // `==` on doubles deliberately. Two platforms that agree to within a tolerance are two
        // platforms that disagree, and the tolerance is where the disagreement would live.
        for (auto index = std::size_t{0}; index < linuxTrace.size(); index++)
        {
            REQUIRE(linuxTrace[index] == windowsTrace[index]);
        }
    }

    SECTION("and the trace is a real one rather than a flat line agreeing with itself")
    {
        const auto [low, high] = std::minmax_element(linuxTrace.begin(), linuxTrace.end());

        REQUIRE(*high - *low > 1.0);
    }
}

TEST_CASE("what the two platforms are allowed to disagree about is the device", "[input][ffb][parity]")
{
    // The other half of criterion 12, and the structural half of criterion 11: the same stage-one
    // newton metres, mapped for two different bases, must come out as two different device commands
    // — because that is where hardware compensation is supposed to live. If these agreed, the
    // mapping stage would not be doing anything and the compensation would have to be somewhere it
    // must never be.
    constexpr auto rackTorque = 12.0;

    auto belt = DeviceForceProfile{};
    belt.peakTorque = 8.0;
    belt.quantisationBits = 8;

    auto directDrive = DeviceForceProfile{};
    directDrive.peakTorque = 20.0;
    directDrive.quantisationBits = 16;

    auto mapping = ForceMapping{};
    mapping.slewRate = 1e9;
    mapping.ceilingTorque = 20.0;

    const auto onBelt = mapRackTorque(belt, mapping, rackTorque, 0.0, rackTorque, 0.002, 1.0);
    const auto onDirect = mapRackTorque(directDrive, mapping, rackTorque, 0.0, rackTorque, 0.002, 1.0);

    // The belt base cannot make twelve newton metres and says so by clipping; the direct drive can.
    REQUIRE(onBelt.commandedTorque < onDirect.commandedTorque);
    REQUIRE(onBelt.commandedTorque <= belt.peakTorque);
    REQUIRE(onDirect.commandedTorque == Catch::Approx(rackTorque));

    // And both were asked for exactly the same thing, which is the invariant that matters.
    REQUIRE(onBelt.requestedTorque == onDirect.requestedTorque);

    // A sixteen-bit device resolves what an eight-bit one cannot, and that difference is a property
    // of the profile rather than of anything upstream.
    REQUIRE(raceengine::quantisationStep(directDrive) < raceengine::quantisationStep(belt));
}

TEST_CASE("a torque fraction reaches DirectInput's magnitude scale intact", "[input][ffb][parity]")
{
    // DirectInput states a constant force in units of ten thousand, signed. Off by a factor here is
    // a Windows build whose wheel is ten times too light or hard against its stops, and neither
    // shows up anywhere but on the hardware.
    REQUIRE(directInputMagnitude(0.0) == 0);
    REQUIRE(directInputMagnitude(1.0) == 10000);
    REQUIRE(directInputMagnitude(-1.0) == -10000);
    REQUIRE(directInputMagnitude(0.5) == 5000);

    // And it is clamped rather than wrapped, because a fraction over one is a bug upstream and the
    // answer to it is full scale, not the other stop.
    REQUIRE(directInputMagnitude(4.0) == 10000);
    REQUIRE(directInputMagnitude(-4.0) == -10000);
}
