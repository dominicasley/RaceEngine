// PresenterService bodies. Declarations are in Graphics/Services/PresenterService.cppm.
//
// A **module implementation unit** — `module raceengine.graphics;` with no `export` — which produces an object
// and no BMI, so nothing imports it and nothing rebuilds when it changes. A definition left in the
// interface is part of that module's BMI instead, and editing one rebuilds every importer of
// `raceengine`. Measurements and the rule: docs/build-times.md.
module;

#include <optional>

module raceengine.graphics;

import :PresenterService;
import :IFrameRecorder;
import raceengine.graphics.models;

namespace raceengine
{

PresenterService::PresenterService(IFrameRecorder& frameRecorder) :
    frameRecorder(frameRecorder)
{
}

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
