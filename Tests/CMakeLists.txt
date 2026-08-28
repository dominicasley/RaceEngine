# The engine's unit tests. Nothing ctest runs opens a device, a window or a display: the services
# under test take their seams by reference, so a recording fake stands in for the backend and the
# tests run anywhere. The GPU gates are the shell scripts in scripts/, and they are untouched by
# this.
#
# InputDeviceProbeTests is the one exception and is hidden rather than excluded: its tag begins with
# a dot, so Catch leaves it out of --list-tests and catch_discover_tests never registers it. It is
# run by hand — `./EngineTests "[device]"` — because what it reports depends on what is plugged in,
# and a suite whose count changes when somebody switches a wheel off is a suite nobody trusts.
find_package(Catch2 3 CONFIG REQUIRED)

add_executable(EngineTests
        AccessorUtilityTests.cpp
        AntiRollBarSweepProbe.cpp
        AntiStallTests.cpp
        AntilockBrakingTests.cpp
        AssistDiagnosticProbe.cpp
        BackendParityTests.cpp
        BodyAttitudeProbe.cpp
        BrakeHardwareTests.cpp
        BrakeModelProbe.cpp
        BrakeThermalProbe.cpp
        BrakeThermalTests.cpp
        BrakingUtilisationProbe.cpp
        CameraServiceTests.cpp
        CarAudioTests.cpp
        ClutchTests.cpp
        ComplianceTests.cpp
        ContactPatchChamferProbe.cpp
        ContactPatchTests.cpp
        CreepDiagnosticProbe.cpp
        CreepTests.cpp
        CriteriaGripSweepProbe.cpp
        CouplingTests.cpp
        DamperFrictionProbe.cpp
        DifferentialTests.cpp
        DrivelinePathProbe.cpp
        DrivelineTests.cpp
        EngineCurveValidationProbe.cpp
        EngineLayerTests.cpp
        EnvelopeCandidateProbe.cpp
        EnvelopeResolutionProbe.cpp
        EnvelopeSolverProbe.cpp
        ForceFeedbackTests.cpp
        FrameDiagnosticsTests.cpp
        FrontCamberValidationProbe.cpp
        FrontHardpointStudyProbe.cpp
        FrameInvarianceTests.cpp
        FrustumTests.cpp
        GradeTests.cpp
        GltfColourSpaceTests.cpp
        GltfNodeTransformTests.cpp
        GolfGtiTests.cpp
        HandlingTests.cpp
        InGearProbe.cpp
        InputDeviceProbeTests.cpp
        InstantAxisCheckProbe.cpp
        InputServiceTests.cpp
        InputTests.cpp
        KerbDiscontinuityProbe.cpp
        LaunchDiagnosticProbe.cpp
        JoltRuntimeTests.cpp
        LookupTableTests.cpp
        MemoryStorageTests.cpp
        PhysicalCameraTests.cpp
        PedalCueProbe.cpp
        PedalFeedbackTests.cpp
        PhysicsWorldTests.cpp
        PostProcessingTests.cpp
        ProvingGroundTests.cpp
        PublishedGeometryTests.cpp
        RackTorqueTests.cpp
        RackSignTests.cpp
        RackTraceCornerTests.cpp
        RendererSeamTests.cpp
        RigidBodyTests.cpp
        SceneManagerServiceTests.cpp
        SetupFileTests.cpp
        ShiftTests.cpp
        StalledChatterProbe.cpp
        SteeringGeometryProbe.cpp
        SteeringShimmyProbe.cpp
        SpringDamperKinematicsProbe.cpp
        SuspensionElementTests.cpp
        SuspensionFidelityTests.cpp
        SuspensionLoadPathProbe.cpp
        SuspensionLoadPathTests.cpp
        SuspensionSweepProbe.cpp
        SuspensionTests.cpp
        TelemetryTests.cpp
        TelemetryUnitsProbe.cpp
        TorqueConverterTests.cpp
        TractionControlTests.cpp
        TyreGripRatioSweepProbe.cpp
        TyreLayerTests.cpp
        TyrePeakToTailProbe.cpp
        TyreLoadSensitivityTests.cpp
        TyreShapeProbe.cpp
        TyreTests.cpp
        TyreThermalProbe.cpp
        TyrePressureTests.cpp
        TyreThermalTests.cpp
        VehicleDataProbe.cpp
        AssistPlacementProbe.cpp
        VehicleDeltaProbe.cpp
        VehicleTests.cpp
        WheelSpeedSensorTests.cpp
        WheelThermalTests.cpp
        SphericalHarmonicsTests.cpp
        ShadowCascadesTests.cpp
        VolumetricFogTests.cpp)

# The recording fake is a module unit rather than a header because the seams it implements are
# only reachable through `import`, and a header may not import.
target_sources(EngineTests PRIVATE
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES CapturedLog.cppm RecordingRenderBackend.cppm RecordingInputBackend.cppm)

target_link_libraries(EngineTests PRIVATE Engine)
target_link_libraries(EngineTests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(EngineTests PRIVATE raceengine_warnings)

# The precompiled header, on the one target that can take it. This is the largest target in the
# tree by compile time — 536 s of the build's serial total — and its sixty-odd plain translation
# units all import the engine and include the standard headers the PCH carries, which is exactly
# the combination the PCH removes the cost of.
#
# **The three module units above are excluded, and must stay excluded**: clang-19 segfaults in the
# frontend building a module interface unit against a PCH, which is why neither Engine nor
# OpenStreetRace has one. A plain translation unit is fine. Full account: docs/build-times.md.
set_source_files_properties(CapturedLog.cppm RecordingRenderBackend.cppm RecordingInputBackend.cppm
        PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
target_precompile_headers(EngineTests PRIVATE ${CMAKE_SOURCE_DIR}/RaceEngine/cmake/raceengine_pch.hpp)

# One CTest entry per TEST_CASE, so a failure names the case rather than the binary. The
# enumeration runs the freshly built binary, which is itself a check that it starts headless.
include(Catch)
catch_discover_tests(EngineTests)
