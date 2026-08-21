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
        AntiStallTests.cpp
        BackendParityTests.cpp
        CameraServiceTests.cpp
        CarAudioTests.cpp
        ClutchTests.cpp
        ComplianceTests.cpp
        ContactPatchTests.cpp
        CouplingTests.cpp
        DifferentialTests.cpp
        DrivelineTests.cpp
        EngineLayerTests.cpp
        ForceFeedbackTests.cpp
        FrameDiagnosticsTests.cpp
        FrustumTests.cpp
        GltfColourSpaceTests.cpp
        GltfNodeTransformTests.cpp
        GolfGtiTests.cpp
        HandlingTests.cpp
        InputDeviceProbeTests.cpp
        InputServiceTests.cpp
        InputTests.cpp
        JoltRuntimeTests.cpp
        LookupTableTests.cpp
        MemoryStorageTests.cpp
        PhysicalCameraTests.cpp
        PhysicsWorldTests.cpp
        PostProcessingTests.cpp
        ProvingGroundTests.cpp
        PublishedGeometryTests.cpp
        RackTorqueTests.cpp
        RackSignTests.cpp
        RendererSeamTests.cpp
        RigidBodyTests.cpp
        SceneManagerServiceTests.cpp
        SetupFileTests.cpp
        ShiftTests.cpp
        SteeringShimmyProbe.cpp
        SuspensionTests.cpp
        TelemetryTests.cpp
        TorqueConverterTests.cpp
        TyreLayerTests.cpp
        TyreTests.cpp
        VehicleTests.cpp
        SphericalHarmonicsTests.cpp
        ShadowCascadesTests.cpp)

# The recording fake is a module unit rather than a header because the seams it implements are
# only reachable through `import`, and a header may not import.
target_sources(EngineTests PRIVATE
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES CapturedLog.cppm RecordingRenderBackend.cppm RecordingInputBackend.cppm)

target_link_libraries(EngineTests PRIVATE Engine)
target_link_libraries(EngineTests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(EngineTests PRIVATE raceengine_warnings)

# One CTest entry per TEST_CASE, so a failure names the case rather than the binary. The
# enumeration runs the freshly built binary, which is itself a check that it starts headless.
include(Catch)
catch_discover_tests(EngineTests)
