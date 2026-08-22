// SoundBank bodies. Declarations are in Audio/Api/SoundBank.cppm.
//
// A **module implementation unit** — `module raceengine.audio;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

module raceengine.audio;

import :SoundBank;

namespace raceengine
{

namespace
{

[[nodiscard]] int hexDigit(const char character)
{
    if (character >= '0' && character <= '9')
    {
        return character - '0';
    }

    if (character >= 'a' && character <= 'f')
    {
        return character - 'a' + 10;
    }

    if (character >= 'A' && character <= 'F')
    {
        return character - 'A' + 10;
    }

    return -1;
}

[[nodiscard]] bool parseHex(const std::string_view text, std::uint64_t& value)
{
    value = 0;

    if (text.empty())
    {
        return false;
    }

    for (const auto character : text)
    {
        const auto digit = hexDigit(character);
        if (digit < 0)
        {
            return false;
        }

        value = (value << 4) | static_cast<std::uint64_t>(digit);
    }

    return true;
}

} // namespace

[[nodiscard]] std::expected<SoundGuid, std::string> parseSoundGuid(std::string_view text)
{
    if (text.size() >= 2 && text.front() == '{' && text.back() == '}')
    {
        text = text.substr(1, text.size() - 2);
    }

    if (text.size() != 36 || text[8] != '-' || text[13] != '-' || text[18] != '-' || text[23] != '-')
    {
        return std::unexpected("not a GUID: '" + std::string(text) + "'");
    }

    auto first = std::uint64_t{};
    auto second = std::uint64_t{};
    auto third = std::uint64_t{};

    if (!parseHex(text.substr(0, 8), first) || !parseHex(text.substr(9, 4), second) ||
        !parseHex(text.substr(14, 4), third))
    {
        return std::unexpected("not a GUID: '" + std::string(text) + "'");
    }

    auto guid = SoundGuid{};
    guid.data1 = static_cast<std::uint32_t>(first);
    guid.data2 = static_cast<std::uint16_t>(second);
    guid.data3 = static_cast<std::uint16_t>(third);

    // The last two groups are bytes in the order written, not a number: byte four of the fourth field
    // is the *fifth* character pair, and reading the pair as an integer reverses them.
    const auto tail = std::string(text.substr(19, 4)) + std::string(text.substr(24, 12));

    for (auto index = std::size_t{0}; index < 8; index++)
    {
        auto byte = std::uint64_t{};
        if (!parseHex(std::string_view(tail).substr(index * 2, 2), byte))
        {
            return std::unexpected("not a GUID: '" + std::string(text) + "'");
        }

        guid.data4[index] = static_cast<std::uint8_t>(byte);
    }

    return guid;
}

[[nodiscard]] std::expected<SoundBankMap, std::string> parseSoundBankMap(const std::string_view text,
                                                                         const std::string_view bankPath)
{
    auto map = SoundBankMap{};
    map.bankPath = std::string(bankPath);

    auto remaining = text;
    auto lineNumber = std::size_t{0};

    while (!remaining.empty())
    {
        lineNumber++;

        const auto breakAt = remaining.find('\n');
        auto line = breakAt == std::string_view::npos ? remaining : remaining.substr(0, breakAt);
        remaining = breakAt == std::string_view::npos ? std::string_view{} : remaining.substr(breakAt + 1);

        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
        {
            line.remove_suffix(1);
        }

        if (line.empty())
        {
            continue;
        }

        const auto gap = line.find(' ');
        if (gap == std::string_view::npos)
        {
            continue;
        }

        const auto guid = parseSoundGuid(line.substr(0, gap));
        if (!guid)
        {
            return std::unexpected("line " + std::to_string(lineNumber) + ": " + guid.error());
        }

        const auto path = line.substr(gap + 1);

        if (path.starts_with("bank:/"))
        {
            // The first bank line is the car's own; `bank:/common` is AC's shared one and is not this.
            if (!path.ends_with("/common") && !map.bank.valid())
            {
                map.bank = guid.value();
            }

            continue;
        }

        if (!path.starts_with("event:/"))
        {
            continue;
        }

        const auto slash = path.rfind('/');
        const auto role = slash == std::string_view::npos ? path : path.substr(slash + 1);

        for (auto index = std::size_t{0}; index < static_cast<std::size_t>(CarEvent::Count); index++)
        {
            if (role == carEventName(static_cast<CarEvent>(index)))
            {
                map.events[index] = guid.value();
                break;
            }
        }
    }

    if (!map.bank.valid())
    {
        return std::unexpected("no bank: line in this GUID map, so nothing says which bank to load");
    }

    return map;
}

} // namespace raceengine
