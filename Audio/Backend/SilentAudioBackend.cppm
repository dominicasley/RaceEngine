module;

#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

module raceengine.audio:SilentAudioBackend;

import :AudioBackend;
import :CarAudio;
import :SoundBank;

namespace raceengine
{

// No device, no sound, no refusals.
//
// This is what an unattended run uses and what a machine with no FMOD falls back to, and it is a real
// object rather than a branch around the audio path on purpose: the gates capture frames with the
// game fully wired, and the only way to be sure audio changes no pixel is for the silent case to run
// the same code every other case runs and simply make nothing of it.
class SilentAudioBackend final : public IAudioBackend
{
    bool loaded = false;

public:
    [[nodiscard]] std::string_view platform() const override
    {
        return "silent";
    }

    [[nodiscard]] std::expected<void, std::string> loadCar(const SoundBankMap&, double, double) override
    {
        loaded = true;

        return {};
    }

    void update(const CarAudioState&) override
    {
    }

    void unloadCar() override
    {
        loaded = false;
    }

    [[nodiscard]] std::vector<std::string> unmatchedParameters() const override
    {
        return {};
    }

    [[nodiscard]] std::vector<std::string> declaredParameters() const override
    {
        return {};
    }
};

std::unique_ptr<IAudioBackend> createSilentAudioBackend();

} // namespace raceengine
