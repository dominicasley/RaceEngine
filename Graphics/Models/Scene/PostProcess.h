#pragma once

#include <vector>

#include "Shader.h"
#include "Texture.h"
#include "Fbo.h"

#include <Shared/Types/Resource.h>

struct PostProcess {
    Resource<Shader> shader;
    std::optional<Resource<Fbo>> output;
    std::vector<Resource<FboAttachment>> inputs;
    std::vector<Resource<FboAttachment>> attachments;
};