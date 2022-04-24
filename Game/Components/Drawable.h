#pragma once

#include <string>

#include "Graphics/Graphics.h"
#include "Game/Models/Component.h"

struct Drawable : public Component {
public:
    RenderableEntity* renderableEntity;
};