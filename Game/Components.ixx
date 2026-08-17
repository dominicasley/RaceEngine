module;

#include <string>
#include "Graphics/Graphics.h"

export module Game:Components;

import :Models;

export struct Drawable : public Component
{
public:
    RenderableEntity& renderableEntity;
    std::optional<std::function<void()>> beforeDraw;
    std::optional<std::function<void()>> afterDraw;

    explicit Drawable(RenderableEntity& _renderableEntity) : renderableEntity(_renderableEntity)
    {

    }
};