module;

#include <optional>

#include <spdlog/logger.h>

export module raceengine.graphics:PresenterService;

import :OpenGLRenderer;
import raceengine.graphics.models;

namespace raceengine
{

export class PresenterService
{
private:
    spdlog::logger& logger;
    OpenGLRenderer& renderer;
    std::optional<Presenter> presenter;

public:
    explicit PresenterService(spdlog::logger& logger, OpenGLRenderer& renderer);
    void setPresenter(const Presenter& presenter);
    void present() const;

};

PresenterService::PresenterService(spdlog::logger& logger, OpenGLRenderer& renderer) : logger(logger),
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
