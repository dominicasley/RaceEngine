// ColourGradeService bodies. Declarations are in Graphics/Services/ColourGradeService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <cstring>
#include <expected>
#include <string>
#include <string_view>

#include <spdlog/logger.h>

module raceengine.graphics;

import :ColourGradeService;
import :LookupTable;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

ColourGradeService::ColourGradeService(spdlog::logger& logger, MemoryStorageService& memoryStorageService) :
    logger(logger),
    memoryStorageService(memoryStorageService)
{
}

std::expected<Resource<Texture>, std::string> ColourGradeService::load(const std::string& name,
                                                                       const std::string_view source) const
{
    const auto table = parseCubeLookupTable(source);
    if (!table)
    {
        return std::unexpected("colour grade '" + name + "' did not parse: " + table.error());
    }

    if (table->domainMinimum != 0.0f || table->domainMaximum != 1.0f)
    {
        // The sampling in the present shader maps 0..1 onto the cube. A table over another domain
        // is a log-encoded grade, which would need its own transfer applied first — refused here
        // rather than sampled as though it were the display range, because that reads as a grade
        // that is merely wrong.
        return std::unexpected("colour grade '" + name + "' is defined over " + std::to_string(table->domainMinimum) +
                               ".." + std::to_string(table->domainMaximum) + "; this engine samples tables over 0..1");
    }

    // Three channels at float precision, as the file states them. The backend pads to four on the
    // way to the image, exactly as it does for an RGB picture.
    auto texture = Texture{.name = name,
                           .format = TextureFormat::RGB,
                           .pixelDataType = PixelDataType::Float,
                           .width = table->size,
                           .height = table->size,
                           .depth = table->size,
                           .bitsPerPixel = 96,
                           .data = {}};
    texture.data.resize(table->entries.size() * sizeof(float));
    std::memcpy(texture.data.data(), table->entries.data(), texture.data.size());

    logger.info("Colour grade '{}' loaded: {} cube, {} entries", name, table->size, table->entries.size() / 3);

    return memoryStorageService.textures.add(std::move(texture));
}

std::expected<Resource<Texture>, std::string> ColourGradeService::loadStrip(const std::string& name,
                                                                            const Resource<Texture>& strip) const
{
    const auto* source = memoryStorageService.textures.find(strip);
    if (source == nullptr)
    {
        return std::unexpected("colour grade '" + name + "' names a strip that is no longer loaded");
    }

    const auto size = source->height;
    if (size < 2 || size > maximumLookupTableSize || source->width != size * size)
    {
        return std::unexpected("colour grade '" + name + "' is " + std::to_string(source->width) + "x" +
                               std::to_string(source->height) + "; a strip is N slices of N squared, so its width " +
                               "must be the square of its height");
    }

    const auto channels = static_cast<size_t>(source->format == TextureFormat::RGBA  ? 4
                                              : source->format == TextureFormat::RGB ? 3
                                                                                     : 0);
    if (channels == 0 || source->pixelDataType != PixelDataType::UnsignedByte)
    {
        return std::unexpected("colour grade '" + name + "' must be an eight-bit RGB or RGBA image");
    }

    if (source->data.size() < static_cast<size_t>(source->width) * source->height * channels)
    {
        return std::unexpected("colour grade '" + name + "' carries less pixel data than its size needs");
    }

    // No transfer function on the way in, deliberately. The table's values *are* display values —
    // it was authored by grading a picture of a screen — and the pass that reads it runs on an
    // image the tone map has already made display-referred. Decoding it as sRGB here would grade
    // the frame with a table that had been linearised out from under it.
    LookupTable table{.size = size, .domainMinimum = 0.0f, .domainMaximum = 1.0f, .entries = {}};
    table.entries.reserve(static_cast<size_t>(size) * size * size * 3);

    for (auto blue = 0u; blue < size; blue++)
    {
        for (auto green = 0u; green < size; green++)
        {
            for (auto red = 0u; red < size; red++)
            {
                const auto x = blue * size + red;
                const auto texel = (static_cast<size_t>(green) * source->width + x) * channels;

                for (size_t channel = 0; channel < 3; channel++)
                {
                    table.entries.push_back(static_cast<float>(source->data[texel + channel]) / 255.0f);
                }
            }
        }
    }

    auto texture = Texture{.name = name,
                           .format = TextureFormat::RGB,
                           .pixelDataType = PixelDataType::Float,
                           .width = size,
                           .height = size,
                           .depth = size,
                           .bitsPerPixel = 96,
                           .data = {}};
    texture.data.resize(table.entries.size() * sizeof(float));
    std::memcpy(texture.data.data(), table.entries.data(), texture.data.size());

    const auto graded = memoryStorageService.textures.add(std::move(texture));
    memoryStorageService.textures.remove(strip);

    logger.info("Colour grade '{}' unrolled from a {}x{} strip: {} cube", name, size * size, size, size);

    return graded;
}

} // namespace raceengine
