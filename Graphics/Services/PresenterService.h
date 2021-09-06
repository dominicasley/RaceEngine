#pragma once


#include <spdlog/logger.h>
#include <optional>

#include "Graphics/Api/OpenGLRenderer.h"
#include "Graphics/Models/Scene/Presenter.h"

class PresenterService
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


