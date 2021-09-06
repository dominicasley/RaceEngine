#pragma once

#include <optional>

#include "Texture.h"
#include "TextureFormat.h"
#include "FboAttachmentType.h"

struct FboAttachment
{
    FboAttachmentType type;
    std::optional<unsigned int> gpuResourceId;
    unsigned int width;
    unsigned int height;
    TextureFormat captureFormat;
    TextureFormat internalFormat;
    std::vector<unsigned char> data;
};
