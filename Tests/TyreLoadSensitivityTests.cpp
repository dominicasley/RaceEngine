#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>

import raceengine.physics;

using raceengine::bringUpJolt;
using raceengine::cornerCount;
using raceengine::generateProvingGround;
using raceengine::golfGtiMk7;
using raceengine::noDriveTorque;
using raceengine::PhysicsWorld;
using raceengine::placeholderSedan;
using raceengine::ProvingGroundDescriptor;
using raceengine::stepVehicle;
using raceengine::tearDownJolt;
using raceengine::TyreModel;
using raceengine::VehicleInput;
using raceengine::VehicleSetup;
using raceengine::VehicleState;

// The load-sensitivity exponent, after it was split in two. `docs/tyre-grip-ratio-brief.md`, stage 1.
//
// AC's `tyres.ini` states **two** exponents — `LS_EXP_Y` and `LS_EXP_X` — and this model carried one,
// used the lateral, and threw the longitudinal away. The comment in `PublishedCarsImpl` said so out
// loud. Splitting it is a faithfulness fix rather than a tune: the model was discarding a number its
// own data source states.
//
// The first case here is the *inertness* proof and it is the reason this file exists. A restructure
// that claims to move nothing has to show it rather than say it — the standing rule after a moved
// golden was blessed on an assertion — so the placeholder car, which sets neither exponent and
// therefore takes both defaults, is driven through a manoeuvre with lateral *and* longitudinal
// content and its final state is compared byte for byte against a capture taken on the build before
// the split.

namespace
{

constexpr auto tick = 1.0 / 360.0;

struct JoltGuard
{
    JoltGuard()
    {
        REQUIRE(bringUpJolt().has_value());
    }

    JoltGuard(const JoltGuard&) = delete;
    JoltGuard& operator=(const JoltGuard&) = delete;

    ~JoltGuard()
    {
        tearDownJolt();
    }
};

[[nodiscard]] ProvingGroundDescriptor plate()
{
    auto descriptor = ProvingGroundDescriptor{};
    descriptor.length = 400.0;
    descriptor.width = 60.0;
    descriptor.cellSize = 2.0;
    descriptor.features = {};

    return descriptor;
}

// FNV-1a over the state's bytes. `VehicleState` is trivially copyable and pinned as such in
// `Vehicle.cppm`, so its object representation is the whole of what the car ended up doing —
// chassis, corner deflections, tyre carcass states and all.
// **Over the state's named numbers and not over its bytes** (2026-08-27). It was a memcpy of the
// whole `VehicleState`, which meant that adding *any* field to `CornerState` moved the hash while
// every physical number stayed bit-identical — and this case exists to say whether the physics
// moved, so a layout change reading as a physics change is the one failure it must not have. That
// happened the day `complianceSteer` was added, and the four printed doubles are what told the two
// apart, exactly as the note below said they would.
//
// Fed field by field in a fixed order instead. A field added to the state no longer disturbs it; a
// field whose *value* changes still does.
[[nodiscard]] std::uint64_t hashOf(const VehicleState& state)
{
    auto hash = std::uint64_t{0xcbf29ce484222325ULL};

    const auto feed = [&hash](const double value)
    {
        auto bytes = std::array<unsigned char, sizeof(double)>{};
        std::memcpy(bytes.data(), &value, sizeof(double));

        for (const auto byte : bytes)
        {
            hash ^= byte;
            hash *= 0x100000001b3ULL;
        }
    };

    feed(state.chassis.position.x);
    feed(state.chassis.position.y);
    feed(state.chassis.position.z);
    feed(state.chassis.orientation.w);
    feed(state.chassis.orientation.x);
    feed(state.chassis.orientation.y);
    feed(state.chassis.orientation.z);
    feed(state.chassis.linearVelocity.x);
    feed(state.chassis.linearVelocity.y);
    feed(state.chassis.linearVelocity.z);
    feed(state.chassis.angularMomentum.x);
    feed(state.chassis.angularMomentum.y);
    feed(state.chassis.angularMomentum.z);

    for (const auto& corner : state.corners)
    {
        feed(corner.wishboneAngle);
        feed(corner.wishboneRate);
        feed(corner.wheelSpeed);
    }

    return hash;
}

// Settle, roll, then steer and brake together. Both axes are exercised on purpose: an exponent that
// reached the wrong axis would be invisible in a straight line and invisible in a steady corner, and
// only combined slip puts both peaks in the same tick.
[[nodiscard]] VehicleState driveFixedManoeuvre(const VehicleSetup& setup, const PhysicsWorld& world)
{
    auto state = VehicleState{};
    state.chassis.position = glm::dvec3(0.0, 0.7, 20.0);

    for (auto step = 0; step < 720; step++)
    {
        REQUIRE(stepVehicle(setup, state, VehicleInput{}, noDriveTorque, world, tick).has_value());
    }

    state.chassis.linearVelocity = glm::dvec3(0.0, 0.0, 25.0);
    for (auto& corner : state.corners)
    {
        corner.wheelSpeed = 25.0 / setup.corners.front().hardpoints.wheelRadius;
    }

    for (auto step = 0; step < 1080; step++)
    {
        auto input = VehicleInput{};
        input.steering = 0.25 * std::sin(static_cast<double>(step) * 0.004);
        input.brake = step > 360 ? 0.30 : 0.0;

        REQUIRE(stepVehicle(setup, state, input, noDriveTorque, world, tick).has_value());
    }

    return state;
}

} // namespace

TEST_CASE("splitting the load-sensitivity exponent moves no car that did not state two", "[physics][tyre][loadsplit]")
{
    // **The capture, and where it came from.** Taken on the build immediately before the split, with
    // `TyreModel` carrying a single `loadSensitivity = 0.15` — the same manoeuvre, the same car, the
    // same 1800 ticks. If the split were to reach a car that states neither exponent, this is the
    // number that would move.
    //
    // The state itself is printed alongside so that a compiler or Jolt change can be told apart from
    // a physics change: a hash that moves says only that something did, and the four numbers below
    // say what. Captured 2026-08-23 on clang-19, build/dev:
    //
    //   z 75.418708063733618   x -21.190494614318158
    //   vz 6.359738717902608   vx -9.892986942826541
    //
    // **This literal is not a golden to re-bless.** It is a record of what the car did before a
    // change that was supposed to reach nothing without two stated exponents. If it moves, either
    // the split leaked or the toolchain did, and the four numbers above are how to tell which.
    //
    // **The hash literal was re-taken once, on 2026-08-27, and the four numbers are why that was
    // safe.** `CornerState` gained a `complianceSteer` field that day; the hash was a memcpy of the
    // whole state, so it moved on the struct's *layout* while every number above stayed identical to
    // the last printed digit. The function now feeds named fields rather than bytes, which is what
    // makes it a statement about the car instead of about the struct — and the four doubles are
    // asserted directly below, so the content claim no longer rests on the hash at all.
    constexpr auto capturedHash = std::uint64_t{0xbc62841291c91f17ULL};

    const JoltGuard jolt;

    const auto setup = placeholderSedan();
    REQUIRE(setup.has_value());

    // The precondition the whole case rests on: this car states no exponent at all, so it is a car
    // the split is supposed to be invisible to. Asserted rather than assumed — if the placeholder
    // ever takes a stated exponent, this case stops proving what it claims to.
    REQUIRE(setup->corners.front().tyre.nominalLoad == TyreModel{}.nominalLoad);

    const auto world = PhysicsWorld::create(generateProvingGround(plate()).value());
    REQUIRE(world.has_value());

    const auto state = driveFixedManoeuvre(setup.value(), world.value());

    std::printf("\n[loadsplit] placeholder final state: hash 0x%016llx  z %.15f  x %.15f  vz %.15f  vx %.15f\n",
                static_cast<unsigned long long>(hashOf(state)), state.chassis.position.z, state.chassis.position.x,
                state.chassis.linearVelocity.z, state.chassis.linearVelocity.x);

    CAPTURE(state.chassis.position.z, state.chassis.position.x, state.chassis.linearVelocity.z);

    // **The content claim, and it is the one that matters.** These are the four numbers recorded on
    // 2026-08-23, asserted to the bit. They do not depend on how `VehicleState` is laid out, on how
    // it is hashed, or on any field anybody adds to it later — so a change that moves them is a
    // change to the car, full stop.
    REQUIRE(state.chassis.position.z == 75.418708063733618);
    REQUIRE(state.chassis.position.x == -21.190494614318158);
    REQUIRE(state.chassis.linearVelocity.z == 6.359738717902608);
    REQUIRE(state.chassis.linearVelocity.x == -9.892986942826541);

    REQUIRE(hashOf(state) == capturedHash);
}

TEST_CASE("each exponent acts on its own axis and on nothing else", "[physics][tyre][loadsplit]")
{
    // The defect the split makes possible, pinned: a peak read off one axis against the other axis's
    // exponent. `tyreFriction` takes the axis rather than the peak precisely so that no call site can
    // spell the mismatch — this is the check that the function itself does not.
    auto model = TyreModel{};
    model.lateralPeak = 1.0;
    model.longitudinalPeak = 1.0;
    model.lateralLoadSensitivity = 0.40;
    model.longitudinalLoadSensitivity = 0.10;

    // Twice nominal load. mu = peak * 2^-k, so the lateral falls to 2^-0.40 = 0.7579 and the
    // longitudinal only to 2^-0.10 = 0.9330. Hand arithmetic, not a recorded run.
    const auto lateral = raceengine::tyreFriction(model, raceengine::TyreAxis::Lateral, 2.0 * model.nominalLoad, 1.0);
    const auto longitudinal =
        raceengine::tyreFriction(model, raceengine::TyreAxis::Longitudinal, 2.0 * model.nominalLoad, 1.0);

    REQUIRE(lateral == Catch::Approx(0.757858283).epsilon(1e-6));
    REQUIRE(longitudinal == Catch::Approx(0.933032992).epsilon(1e-6));

    SECTION("and moving one leaves the other exactly alone")
    {
        auto flattened = model;
        flattened.longitudinalLoadSensitivity = 0.0;

        REQUIRE(raceengine::tyreFriction(flattened, raceengine::TyreAxis::Lateral, 2.0 * model.nominalLoad, 1.0) ==
                lateral);
        REQUIRE(raceengine::tyreFriction(flattened, raceengine::TyreAxis::Longitudinal, 2.0 * model.nominalLoad, 1.0) ==
                Catch::Approx(1.0));
    }
}

TEST_CASE("the imported car states both of the exponents its data file carries", "[physics][golf][loadsplit]")
{
    // Acceptance item 3 of `docs/tyre-grip-ratio-brief.md`: each exponent sourced to the file's own
    // figure. AC states the force as going with load to the LS_EXP power, so this model's exponent is
    // that power less one with the sign turned over.
    const auto setup = golfGtiMk7();
    REQUIRE(setup.has_value());

    for (const auto& corner : setup->corners)
    {
        // tyres.ini LS_EXP_Y = 0.8074, Semislicks.
        REQUIRE(corner.tyre.lateralLoadSensitivity == Catch::Approx(0.1926).margin(1e-9));
        // tyres.ini LS_EXP_X = 0.8756, Semislicks — stated by the file, discarded by this model until
        // 2026-08-23, and the reason this file exists.
        REQUIRE(corner.tyre.longitudinalLoadSensitivity == Catch::Approx(0.1244).margin(1e-9));
    }

    SECTION("and they are different numbers, which is the whole of the finding")
    {
        REQUIRE(setup->corners.front().tyre.longitudinalLoadSensitivity <
                setup->corners.front().tyre.lateralLoadSensitivity);
    }

    SECTION("so a braked front wheel keeps more of its longitudinal peak than the old single exponent left it")
    {
        // A front wheel carries about 5577 N under maximum braking against a nominal 2939 N, so
        // mu_x = 1.131 * (5577/2939)^-k. At the lateral exponent, k = 0.1926 gives **0.9997**; at the
        // file's own longitudinal one, k = 0.1244 gives **1.0444**. Both arithmetic, to four figures.
        //
        // The brief's own hand calculation of this said 0.997 and 1.043 — right to a part in a
        // thousand and quoted here because it is the number a reader will have in front of them.
        const auto& tyre = setup->corners.front().tyre;

        const auto braking = raceengine::tyreFriction(tyre, raceengine::TyreAxis::Longitudinal, 5577.0, 1.0);

        auto asShipped = tyre;
        asShipped.longitudinalLoadSensitivity = tyre.lateralLoadSensitivity;
        const auto before = raceengine::tyreFriction(asShipped, raceengine::TyreAxis::Longitudinal, 5577.0, 1.0);

        REQUIRE(before == Catch::Approx(0.9997).margin(0.0002));
        REQUIRE(braking == Catch::Approx(1.0444).margin(0.0002));

        // 4.5% more longitudinal grip where a stop actually happens, and it costs nothing anywhere:
        // it is the number the data file already carried.
        REQUIRE(braking / before == Catch::Approx(1.0447).margin(0.0005));
    }
}
