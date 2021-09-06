#pragma once

#include <optional>
#include <vector>

#include "FboAttachment.h"
#include "FboType.h"
#include <Shared/Types/Resource.h>

struct Fbo
{
    FboType type;
    std::optional<unsigned int> gpuResourceId;
    std::vector<Resource<FboAttachment>> attachments;
};