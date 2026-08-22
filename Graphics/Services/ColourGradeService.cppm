module;

#include <cstring>
#include <expected>
#include <string>
#include <string_view>

#include <spdlog/logger.h>

export module raceengine.graphics:ColourGradeService;

import :LookupTable;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class ColourGradeService
{
private:
    spdlog::logger& logger;
    MemoryStorageService& memoryStorageService;

public:
    explicit ColourGradeService(spdlog::logger& logger, MemoryStorageService& memoryStorageService);
    // A `.cube` file's text into a table the presenter can name. The upload happens on the frame it
    // is first presented with, like every other texture: this is the parse and the storage, and it
    // reports rather than throws because the file is the user's and a malformed one is a thing to
    // be told about, not a crash.
    [[nodiscard]] std::expected<Resource<Texture>, std::string> load(const std::string& name,
                                                                     std::string_view source) const;
    // The other format a grade arrives in: an unrolled strip, `size` slices of `size` squared laid
    // out in a row — 256x16 for the sixteen-cube everything in the Unreal ecosystem publishes.
    // Blue picks the slice, red runs across it and green down it.
    //
    // Worth carrying beside `.cube` because it is what a decade of forum posts and marketplace
    // packs actually contain, and because it is the format you can author by grading a *picture*:
    // drop the neutral strip into a photo of your game, grade the photo, and the strip that comes
    // out with it is the grade. It arrives already loaded, because decoding an image is the
    // resource path's job and not this one's; the strip is dropped afterwards, being a delivery
    // format rather than something the frame samples.
    [[nodiscard]] std::expected<Resource<Texture>, std::string> loadStrip(const std::string& name,
                                                                          const Resource<Texture>& strip) const;
};

} // namespace raceengine
