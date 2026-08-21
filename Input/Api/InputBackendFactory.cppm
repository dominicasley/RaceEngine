module;

#include <memory>

export module raceengine.input:InputBackendFactory;

import :InputBackend;

namespace raceengine
{

// The only public way to obtain a backend, and the reason both concrete backends are
// implementation partitions. This unit names IInputBackend and nothing below it, so importing
// raceengine.input costs a consumer the seam and not <linux/input.h> on one platform or Windows.h,
// COM and <dinput.h> on the other.
//
// `nativeWindow` is the platform's own window handle, or null. DirectInput's cooperative level is
// set against a window and force feedback needs the exclusive one, so a Windows build that wants a
// wheel to push back passes its HWND here; every other configuration passes null and is told, as a
// missing capability, what that costs. It is a `void*` because this declaration is read on a
// platform that has no HWND to name.
export [[nodiscard]] std::unique_ptr<IInputBackend> createInputBackend(void* nativeWindow);

} // namespace raceengine
