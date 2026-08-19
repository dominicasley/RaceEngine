module;

#include <optional>

export module raceengine.graphics:PresenterService;

import :IFrameRecorder;
import raceengine.graphics.models;

namespace raceengine
{

export class PresenterService
{
private:
    IFrameRecorder& frameRecorder;
    std::optional<Presenter> presenter;

public:
    explicit PresenterService(IFrameRecorder& frameRecorder);
    void setPresenter(const Presenter& presenter);
    // Records the frame's last pass into the frame Engine::step opened. It does not present:
    // the pass reaches the screen when that step's endFrame submits, which is what keeps one
    // step to one present however many views were recorded before this.
    void record() const;
};

PresenterService::PresenterService(IFrameRecorder& frameRecorder) :
    frameRecorder(frameRecorder)
{
}

// A game that has not chosen a presenter records nothing, and the backend's frame is left to
// put the clear colour on screen. That is a level still being built, not a failure.
void PresenterService::record() const
{
    if (presenter.has_value())
    {
        frameRecorder.recordPresent(presenter.value());
    }
}

void PresenterService::setPresenter(const Presenter& _presenter)
{
    this->presenter = _presenter;
}

} // namespace raceengine
