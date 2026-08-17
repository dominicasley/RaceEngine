module;

#include <optional>

export module raceengine.graphics:PresenterService;

import :IRenderer;
import raceengine.graphics.models;

namespace raceengine
{

export class PresenterService
{
private:
    IRenderer& renderer;
    std::optional<Presenter> presenter;

public:
    explicit PresenterService(IRenderer& renderer);
    void setPresenter(const Presenter& presenter);
    void present() const;
};

PresenterService::PresenterService(IRenderer& renderer) :
    renderer(renderer)
{
}

void PresenterService::present() const
{
    if (presenter.has_value())
    {
        renderer.drawFullScreenQuad(presenter.value().shader, presenter.value().output);
    }
}

void PresenterService::setPresenter(const Presenter& _presenter)
{
    this->presenter = _presenter;
}

} // namespace raceengine
