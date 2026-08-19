module;

#include <memory>

// The one unit in raceengine.input that imports the concrete backends, and an implementation
// partition so that importing it is not something a consumer of the module can do. Everything
// <linux/input.h> and <dinput.h> bring with them stops here.
module raceengine.input:InputBackendFactoryImpl;

import :DirectInputBackend;
import :EvdevInputBackend;
import :InputBackend;
import :InputBackendFactory;

namespace raceengine
{

// Both partitions are imported on both platforms, and that is the point rather than an oversight:
// each one compiles everywhere and only the half of it that is a system call is conditional, so
// the backend that is not shipping today is still built, still warned about and still tested by
// everything in it that does not need the device.
std::unique_ptr<IInputBackend> createInputBackend(void* nativeWindow)
{
#if defined(_WIN32)
    return std::make_unique<DirectInputBackend>(nativeWindow);
#else
    static_cast<void>(nativeWindow);

    return std::make_unique<EvdevInputBackend>();
#endif
}

} // namespace raceengine
