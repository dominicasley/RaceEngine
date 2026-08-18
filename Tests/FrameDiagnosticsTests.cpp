#include <string>

#include <catch2/catch_test_macros.hpp>

import raceengine;
import raceengine.tests.log;

using raceengine::describe;
using raceengine::FrameDiagnostic;
using raceengine::FrameDiagnostics;
using raceengine::tests::CapturedLog;

TEST_CASE("a reason is stated once and its recurrences are counted", "[diagnostics]")
{
    CapturedLog log;
    const std::string reason = describe(FrameDiagnostic::PrimitiveWithoutMaterial);

    {
        FrameDiagnostics diagnostics(log.sink());

        diagnostics.beginFrame();
        diagnostics.record(FrameDiagnostic::PrimitiveWithoutMaterial);
        diagnostics.record(FrameDiagnostic::PrimitiveWithoutMaterial);
        diagnostics.report();

        REQUIRE(log.occurrences(reason) == 1);
        REQUIRE(log.occurrences("2 " + reason) == 1);

        // The second frame meets the same condition. A per-site log-and-continue would print it
        // again — twice — which is exactly what the thirteen `bool …Logged` members this
        // replaced existed to prevent, one bespoke latch at a time.
        diagnostics.beginFrame();
        diagnostics.record(FrameDiagnostic::PrimitiveWithoutMaterial);
        diagnostics.report();

        REQUIRE(log.occurrences(reason) == 1);
        REQUIRE(diagnostics.count(FrameDiagnostic::PrimitiveWithoutMaterial) == 3);
    }

    // The destructor states the totals over the whole run, so a reason that was reported on
    // frame 1 and met 10,000 times after is not silently dropped.
    REQUIRE(log.occurrences("3 " + reason) == 1);
}

TEST_CASE("the detail is built only for a reason's first occurrence", "[diagnostics]")
{
    CapturedLog log;
    FrameDiagnostics diagnostics(log.sink());
    auto detailsBuilt = 0;

    diagnostics.beginFrame();

    for (auto occurrence = 0; occurrence < 5; occurrence++)
    {
        diagnostics.record(FrameDiagnostic::StaleModelHandle,
                           [&]
                           {
                               detailsBuilt++;

                               return "renderable " + std::to_string(occurrence);
                           });
    }

    // Nothing on the draw path may allocate or format per primitive; the branch is the whole
    // cost after the first.
    REQUIRE(detailsBuilt == 1);

    diagnostics.report();

    REQUIRE(log.occurrences("(first: renderable 0)") == 1);
    REQUIRE(log.occurrences("5 " + std::string(describe(FrameDiagnostic::StaleModelHandle))) == 1);
}

TEST_CASE("a frame that skipped nothing says nothing", "[diagnostics]")
{
    CapturedLog log;
    FrameDiagnostics diagnostics(log.sink());

    diagnostics.beginFrame();
    diagnostics.report();
    diagnostics.beginFrame();
    diagnostics.report();

    REQUIRE(log.text().empty());
    REQUIRE(diagnostics.count(FrameDiagnostic::MeshNotUploaded) == 0);
}

TEST_CASE("work skipped before the first frame is carried into the first report", "[diagnostics]")
{
    CapturedLog log;
    FrameDiagnostics diagnostics(log.sink());

    // An upload or a framebuffer creation happens before any frame opens, so its inFrame count is
    // zero when the first report runs and the total is the whole story.
    diagnostics.record(FrameDiagnostic::MeshNotUploaded);
    diagnostics.record(FrameDiagnostic::MeshNotUploaded);

    diagnostics.beginFrame();
    diagnostics.report();

    REQUIRE(log.occurrences("2 " + std::string(describe(FrameDiagnostic::MeshNotUploaded))) == 1);
}

TEST_CASE("a new reason is reported in the frame it first appears", "[diagnostics]")
{
    CapturedLog log;
    FrameDiagnostics diagnostics(log.sink());

    diagnostics.beginFrame();
    diagnostics.record(FrameDiagnostic::JointLimitExceeded);
    diagnostics.report();

    diagnostics.beginFrame();
    diagnostics.record(FrameDiagnostic::JointLimitExceeded);
    diagnostics.record(FrameDiagnostic::LightLimitExceeded);
    diagnostics.record(FrameDiagnostic::LightLimitExceeded);
    diagnostics.report();

    REQUIRE(log.occurrences(describe(FrameDiagnostic::JointLimitExceeded)) == 1);
    REQUIRE(log.occurrences("2 " + std::string(describe(FrameDiagnostic::LightLimitExceeded))) == 1);
    REQUIRE(diagnostics.count(FrameDiagnostic::JointLimitExceeded) == 2);
    REQUIRE(diagnostics.count(FrameDiagnostic::LightLimitExceeded) == 2);
}

TEST_CASE("two tallies in one process keep separate counts", "[diagnostics]")
{
    CapturedLog log;
    FrameDiagnostics first(log.sink());
    FrameDiagnostics second(log.sink());

    first.beginFrame();
    first.record(FrameDiagnostic::SkinningRejected);
    first.record(FrameDiagnostic::SkinningRejected);

    // The point of the tally being a member of Engine rather than a function-local static: two
    // engines in one test runner must not share a count, and a static latch could not say that.
    REQUIRE(first.count(FrameDiagnostic::SkinningRejected) == 2);
    REQUIRE(second.count(FrameDiagnostic::SkinningRejected) == 0);
}
