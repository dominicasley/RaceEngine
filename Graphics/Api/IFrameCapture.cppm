module;

#include <expected>
#include <string>

export module raceengine.graphics:IFrameCapture;

namespace raceengine
{

// Debug readback of the frame that was just presented, written to `path` as a PNG. Its own
// seam on purpose: it is not one of the frame's passes, its only consumer is Engine's
// RACEENGINE_DUMP_FRAME path, and a backend that keeps it out of IFrameRecorder cannot
// smuggle capture work into a pass.
//
// It runs after IFrameRecorder::endFrame has presented, which is the constraint a backend
// has to solve rather than a convenience: a presented image may not be readable, so
// reproducing it can mean rendering the frame's last pass a second time.
export class IFrameCapture
{
public:
    IFrameCapture() = default;
    IFrameCapture(const IFrameCapture&) = delete;
    IFrameCapture(IFrameCapture&&) = delete;
    IFrameCapture& operator=(const IFrameCapture&) = delete;
    IFrameCapture& operator=(IFrameCapture&&) = delete;
    virtual ~IFrameCapture() = default;

    // Reports rather than logging: a capture that silently produced nothing would let a
    // frame gate pass on a file that was never written.
    [[nodiscard]] virtual std::expected<void, std::string> captureFrame(const std::string& path) = 0;
};

} // namespace raceengine
