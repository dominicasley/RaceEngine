#include <string>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

import raceengine;

using Catch::Approx;
using raceengine::identityLookupTable;
using raceengine::maximumLookupTableSize;
using raceengine::parseCubeLookupTable;

namespace
{

// The smallest well-formed grade there is: a two-cube, which is eight entries in red-fastest order.
// Written out in full so the tests below can point at a specific number rather than at a generator.
const std::string twoCube = R"(# a two cube
TITLE "test"
LUT_3D_SIZE 2
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
)";

} // namespace

TEST_CASE("a well-formed cube parses to its own size and entries", "[grade][lut]")
{
    const auto table = parseCubeLookupTable(twoCube);

    REQUIRE(table.has_value());
    REQUIRE(table->size == 2);
    REQUIRE(table->domainMinimum == Approx(0.0f));
    REQUIRE(table->domainMaximum == Approx(1.0f));
    REQUIRE(table->entries.size() == 8 * 3);
    // Red runs fastest, so the second entry is the one with red at its maximum and nothing else.
    REQUIRE(table->entries[3] == Approx(1.0f));
    REQUIRE(table->entries[4] == Approx(0.0f));
    REQUIRE(table->entries[5] == Approx(0.0f));
}

TEST_CASE("comments, blank lines and trailing whitespace are not entries", "[grade][lut]")
{
    auto source = std::string("LUT_3D_SIZE 2\n\n   # nothing here  \n");
    for (auto entry = 0; entry < 8; entry++)
    {
        source += "0.5 0.5 0.5   \n";
    }
    source += "\n# and a trailing comment\n";

    const auto table = parseCubeLookupTable(source);

    REQUIRE(table.has_value());
    REQUIRE(table->entries.size() == 8 * 3);
}

TEST_CASE("a file that is not a 3D lookup table is reported rather than half-read", "[grade][lut][expected]")
{
    REQUIRE_FALSE(parseCubeLookupTable("TITLE \"nothing\"\n").has_value());
    REQUIRE_FALSE(parseCubeLookupTable("LUT_1D_SIZE 16\n0 0 0\n").has_value());
    // Entries before the size means nothing knows how many to expect.
    REQUIRE_FALSE(parseCubeLookupTable("0.0 0.0 0.0\nLUT_3D_SIZE 2\n").has_value());
}

TEST_CASE("a cube with the wrong number of entries is refused", "[grade][lut][expected]")
{
    const auto missing = parseCubeLookupTable("LUT_3D_SIZE 2\n0 0 0\n0 0 0\n");

    REQUIRE_FALSE(missing.has_value());
    // The message names both counts, because the useful thing about a truncated export is how far
    // it got.
    REQUIRE(missing.error().find("8") != std::string::npos);
    REQUIRE(missing.error().find("2") != std::string::npos);
}

TEST_CASE("a size the format could state but nothing should allocate is refused", "[grade][lut][expected]")
{
    REQUIRE_FALSE(parseCubeLookupTable("LUT_3D_SIZE 1\n").has_value());
    REQUIRE_FALSE(
        parseCubeLookupTable("LUT_3D_SIZE " + std::to_string(maximumLookupTableSize + 1) + "\n").has_value());
}

TEST_CASE("a domain whose channels disagree has no representation here", "[grade][lut][expected]")
{
    // A log-encoded grade states its own domain and this engine can carry one — but only one per
    // table, so a file asking for three different ones is refused rather than silently read as red's.
    REQUIRE_FALSE(parseCubeLookupTable("LUT_3D_SIZE 2\nDOMAIN_MAX 1.0 2.0 1.0\n").has_value());
    REQUIRE_FALSE(parseCubeLookupTable("LUT_3D_SIZE 2\nDOMAIN_MIN 0.0\n").has_value());
}

TEST_CASE("an entry that is not a number is reported with its line", "[grade][lut][expected]")
{
    const auto broken = parseCubeLookupTable("LUT_3D_SIZE 2\n0.0 0.0 0.0\n0.0 nan-ish 0.0\n");

    REQUIRE_FALSE(broken.has_value());
    REQUIRE(broken.error().find("line 3") != std::string::npos);
}

TEST_CASE("the identity table maps every corner to itself", "[grade][lut]")
{
    const auto table = identityLookupTable(2);

    REQUIRE(table.size == 2);
    REQUIRE(table.entries.size() == 8 * 3);

    // Red fastest, then green, then blue: entry n has coordinates (n & 1, (n >> 1) & 1, n >> 2) and
    // holds exactly those. This is the layout the upload relies on being a straight copy.
    for (auto entry = 0u; entry < 8u; entry++)
    {
        REQUIRE(table.entries[entry * 3 + 0] == Approx(static_cast<float>(entry & 1u)));
        REQUIRE(table.entries[entry * 3 + 1] == Approx(static_cast<float>((entry >> 1u) & 1u)));
        REQUIRE(table.entries[entry * 3 + 2] == Approx(static_cast<float>(entry >> 2u)));
    }
}

TEST_CASE("the identity table round-trips through the parser", "[grade][lut]")
{
    // What a tool exports and what this engine calls neutral have to be the same table, or a grade
    // authored against the neutral one would arrive shifted.
    const auto identity = identityLookupTable(4);

    std::string source = "LUT_3D_SIZE 4\n";
    for (size_t entry = 0; entry < identity.entries.size(); entry += 3)
    {
        source += std::to_string(identity.entries[entry]) + " " + std::to_string(identity.entries[entry + 1]) + " " +
                  std::to_string(identity.entries[entry + 2]) + "\n";
    }

    const auto parsed = parseCubeLookupTable(source);

    REQUIRE(parsed.has_value());
    REQUIRE(parsed->size == identity.size);
    for (size_t component = 0; component < identity.entries.size(); component++)
    {
        REQUIRE(parsed->entries[component] == Approx(identity.entries[component]).margin(1e-5));
    }
}
