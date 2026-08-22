module;

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

export module raceengine.audio:SoundBank;

namespace raceengine
{

// A car's FMOD bank, as far as anything above the audio backend needs to know it: a file, and the
// events inside it named by what they are rather than by their path.
//
// **A bank on its own is not addressable.** FMOD Studio resolves `event:/cars/.../engine_ext` through
// a companion `<name>.strings.bank`, and an Assetto Corsa car does not ship one — the `sfx` folder is
// the bank and a `GUIDs.txt`. So events are looked up by GUID, and `GUIDs.txt` is the mapping. That
// file is plain text, which makes this parser the whole of what stands between a car folder and a
// playable bank.
//
// Parsed here rather than in the backend for the reason `parseCubeLookupTable` is: a text format with
// a grammar is exactly the thing to pin without a device, and this one decides whether a car makes
// any sound at all.

// The events a car event graph carries, named by role. The paths differ per car — every one of them
// ends `/<car name>/<role>` — so the role is the last path element and is what this keys on.
export enum class CarEvent : std::uint8_t {
    EngineExterior,
    EngineInterior,
    Limiter,
    Turbo,
    Wind,
    Wheel,
    SkidExterior,
    SkidInterior,
    BackfireExterior,
    BackfireInterior,
    TractionControlExterior,
    TractionControlInterior,
    Bodywork,
    Door,
    Horn,
    Count
};

export [[nodiscard]] constexpr std::string_view carEventName(const CarEvent event)
{
    switch (event)
    {
    case CarEvent::EngineExterior:
        return "engine_ext";
    case CarEvent::EngineInterior:
        return "engine_int";
    case CarEvent::Limiter:
        return "limiter";
    case CarEvent::Turbo:
        return "turbo";
    case CarEvent::Wind:
        return "wind";
    case CarEvent::Wheel:
        return "wheel";
    case CarEvent::SkidExterior:
        return "skid_ext";
    case CarEvent::SkidInterior:
        return "skid_int";
    case CarEvent::BackfireExterior:
        return "backfire_ext";
    case CarEvent::BackfireInterior:
        return "backfire_int";
    case CarEvent::TractionControlExterior:
        return "tractioncontrol_ext";
    case CarEvent::TractionControlInterior:
        return "tractioncontrol_int";
    case CarEvent::Bodywork:
        return "bodywork";
    case CarEvent::Door:
        return "door";
    case CarEvent::Horn:
        return "horn";
    case CarEvent::Count:
        break;
    }

    return "";
}

// A GUID as FMOD states it, kept as the sixteen bytes rather than as text so the backend does not
// re-parse a string it was handed. `FMOD_GUID` is four fields in the same order and is assembled from
// this where it is named — which is the only place a Windows-shaped struct may appear.
export struct SoundGuid
{
    std::uint32_t data1 = 0;
    std::uint16_t data2 = 0;
    std::uint16_t data3 = 0;
    std::array<std::uint8_t, 8> data4{};

    [[nodiscard]] bool valid() const
    {
        if (data1 != 0 || data2 != 0 || data3 != 0)
        {
            return true;
        }

        for (const auto byte : data4)
        {
            if (byte != 0)
            {
                return true;
            }
        }

        return false;
    }
};

export struct SoundBankMap
{
    // The bank's own GUID, which is what `loadBankFile` reports back and what a second bank would
    // collide with.
    SoundGuid bank{};
    std::string bankPath;
    std::array<SoundGuid, static_cast<std::size_t>(CarEvent::Count)> events{};

    [[nodiscard]] const SoundGuid& event(const CarEvent which) const
    {
        return events[static_cast<std::size_t>(which)];
    }

    [[nodiscard]] bool has(const CarEvent which) const
    {
        return event(which).valid();
    }
};

// `{xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}` in the layout FMOD writes, which is the Microsoft one: the
// first three fields little endian as numbers, the last eight as bytes in order.
export [[nodiscard]] std::expected<SoundGuid, std::string> parseSoundGuid(std::string_view text);

// A car's `sfx/GUIDs.txt`, which is one `{guid} kind:/path` per line. Buses and VCAs are read past:
// they are the mixer's business and this engine addresses none of them.
//
// An unknown event is skipped rather than refused, because a bank may legitimately carry events this
// engine has no concept of and a car that will not load is worse than a car with no horn. A *missing*
// one is reported by `has()` at the point something tries to play it.
export [[nodiscard]] std::expected<SoundBankMap, std::string> parseSoundBankMap(const std::string_view text,
                                                                                const std::string_view bankPath);

} // namespace raceengine
