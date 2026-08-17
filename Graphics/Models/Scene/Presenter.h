#pragma once

#include <Shared/Types/Resource.h>

#include "FboAttachment.h"
#include "Shader.h"

struct Presenter
{
    Resource<FboAttachment> output;
    Resource<Shader> shader;
};