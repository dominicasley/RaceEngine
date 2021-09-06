#pragma once

#include "Texture.h"
#include "Shader.h"

struct Presenter
{
    Resource<FboAttachment> output;
    Resource<Shader> shader;
};