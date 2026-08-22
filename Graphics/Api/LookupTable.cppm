module;

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

export module raceengine.graphics:LookupTable;

namespace raceengine
{

namespace
{

// Kept here as well as in the implementation unit, and that is legal rather than sloppy: an
// unnamed namespace is internal to each translation unit, so these are two distinct copies
// and not one entity defined twice. This side is needed because an inline or constexpr
// function below calls them, and those cannot move — a caller has to see their bodies.
[[nodiscard]] inline bool parseNumber(const std::string_view token, float& value)
{
    const auto result = std::from_chars(token.data(), token.data() + token.size(), value);

    return result.ec == std::errc{} && result.ptr == token.data() + token.size();
}

// `.cube` is line-oriented, `#` starts a comment, and everything else is whitespace-separated
// tokens. Written out rather than reached for because <sstream> in a global module fragment is a
// heavier dependency than the parse deserves.
[[nodiscard]] inline std::vector<std::string_view> tokenise(const std::string_view line)
{
    std::vector<std::string_view> tokens;
    size_t index = 0;

    while (index < line.size())
    {
        while (index < line.size() && (line[index] == ' ' || line[index] == '\t' || line[index] == '\r'))
        {
            index++;
        }

        if (index >= line.size() || line[index] == '#')
        {
            break;
        }

        const auto start = index;
        while (index < line.size() && line[index] != ' ' && line[index] != '\t' && line[index] != '\r')
        {
            index++;
        }

        tokens.push_back(line.substr(start, index - start));
    }

    return tokens;
}

} // namespace

// A colour grade as a lookup table, in Adobe's `.cube` text format — which is what every grading
// tool exports and therefore the only thing that makes a grade *tunable by whoever is grading it*.
// The alternative is what this replaced: contrast, saturation and a split tone as constants in a
// shader, tuned by editing the shader and rebuilding.
//
// Parsed here rather than in the service that uploads it, and returning data rather than a texture,
// because a text format with a documented grammar is exactly the thing that can be pinned by tests
// without a device — and every failure mode below is a file somebody wrote by hand.

// The largest edge this will accept. 64 is two hundred and sixty thousand texels and beyond what any
// grade needs; the cap is here because the size comes out of the file and a wrong number would
// otherwise be an allocation the size of whatever it said.
export inline constexpr uint32_t maximumLookupTableSize = 64;

export struct LookupTable
{
    // Edge length of the cube: the table holds size³ entries.
    uint32_t size = 0;
    // The input range the table is defined over, which is almost always 0..1 and is stated in the
    // file because a log-encoded grade is not.
    float domainMinimum = 0.0f;
    float domainMaximum = 1.0f;
    // RGB triples, red fastest — the order the format writes them and the order a 3D texture wants
    // its rows in, so the upload is a memcpy and not a transpose.
    std::vector<float> entries;
};

export [[nodiscard]] inline std::expected<LookupTable, std::string> parseCubeLookupTable(const std::string_view source)
{
    LookupTable table{};
    size_t lineNumber = 0;
    size_t offset = 0;

    while (offset <= source.size())
    {
        const auto end = source.find('\n', offset);
        const auto line = source.substr(offset, end == std::string_view::npos ? std::string_view::npos : end - offset);
        offset = end == std::string_view::npos ? source.size() + 1 : end + 1;
        lineNumber++;

        const auto tokens = tokenise(line);
        if (tokens.empty())
        {
            continue;
        }

        const auto at = [&]
        {
            return " on line " + std::to_string(lineNumber);
        };

        if (tokens[0] == "TITLE" || tokens[0] == "LUT_1D_SIZE")
        {
            if (tokens[0] == "LUT_1D_SIZE")
            {
                return std::unexpected("this is a one-dimensional lookup table; a colour grade needs a 3D one" + at());
            }

            continue;
        }

        if (tokens[0] == "LUT_3D_SIZE")
        {
            float size = 0.0f;
            if (tokens.size() < 2 || !parseNumber(tokens[1], size) || size < 2.0f ||
                size > static_cast<float>(maximumLookupTableSize))
            {
                return std::unexpected("LUT_3D_SIZE must be between 2 and " + std::to_string(maximumLookupTableSize) +
                                       at());
            }

            table.size = static_cast<uint32_t>(size);
            table.entries.reserve(static_cast<size_t>(table.size) * table.size * table.size * 3);
            continue;
        }

        if (tokens[0] == "DOMAIN_MIN" || tokens[0] == "DOMAIN_MAX")
        {
            // Stated as a triple in the format and as one number here: a grade whose three channels
            // had different domains is not something this engine's sampling could express, and
            // silently keeping the red one would be worse than saying so.
            float first = 0.0f;
            if (tokens.size() < 4 || !parseNumber(tokens[1], first))
            {
                return std::unexpected(std::string(tokens[0]) + " takes three numbers" + at());
            }

            for (size_t channel = 2; channel < 4; channel++)
            {
                float other = 0.0f;
                if (!parseNumber(tokens[channel], other) || other != first)
                {
                    return std::unexpected(std::string(tokens[0]) + "'s three channels must agree" + at());
                }
            }

            (tokens[0] == "DOMAIN_MIN" ? table.domainMinimum : table.domainMaximum) = first;
            continue;
        }

        // Anything else is an entry, and an entry is three numbers.
        if (tokens.size() != 3)
        {
            return std::unexpected("expected a keyword or three numbers, found " + std::to_string(tokens.size()) +
                                   " token(s)" + at());
        }

        if (table.size == 0)
        {
            return std::unexpected("entries before LUT_3D_SIZE" + at());
        }

        for (const auto& token : tokens)
        {
            float component = 0.0f;
            if (!parseNumber(token, component))
            {
                return std::unexpected("'" + std::string(token) + "' is not a number" + at());
            }

            table.entries.push_back(component);
        }
    }

    if (table.size == 0)
    {
        return std::unexpected("no LUT_3D_SIZE, so this is not a 3D lookup table");
    }

    const auto expected = static_cast<size_t>(table.size) * table.size * table.size * 3;
    if (table.entries.size() != expected)
    {
        return std::unexpected("a " + std::to_string(table.size) + " cube needs " + std::to_string(expected / 3) +
                               " entries, found " + std::to_string(table.entries.size() / 3));
    }

    if (table.domainMaximum <= table.domainMinimum)
    {
        return std::unexpected("DOMAIN_MAX must be above DOMAIN_MIN");
    }

    return table;
}

// The table that changes nothing, which is what a pass with no grade of its own binds: a sampler a
// pipeline statically uses has to be written, and the neutral answer is the identity.
export [[nodiscard]] inline LookupTable identityLookupTable(const uint32_t size)
{
    LookupTable table{.size = std::max(size, 2u), .domainMinimum = 0.0f, .domainMaximum = 1.0f, .entries = {}};
    table.entries.reserve(static_cast<size_t>(table.size) * table.size * table.size * 3);

    const auto last = static_cast<float>(table.size - 1);
    for (auto blue = 0u; blue < table.size; blue++)
    {
        for (auto green = 0u; green < table.size; green++)
        {
            for (auto red = 0u; red < table.size; red++)
            {
                table.entries.push_back(static_cast<float>(red) / last);
                table.entries.push_back(static_cast<float>(green) / last);
                table.entries.push_back(static_cast<float>(blue) / last);
            }
        }
    }

    return table;
}

} // namespace raceengine
