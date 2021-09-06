
#include "PresenterService.h"

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


