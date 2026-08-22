// SilentAudioBackend bodies. Declarations are in Audio/Backend/SilentAudioBackend.cppm.
//
// A **module implementation unit** — `module raceengine.audio;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

module raceengine.audio;

import :SilentAudioBackend;
import :AudioBackend;
import :CarAudio;
import :SoundBank;

namespace raceengine
{

std::unique_ptr<IAudioBackend> createSilentAudioBackend()
{
    return std::make_unique<SilentAudioBackend>();
}

} // namespace raceengine
