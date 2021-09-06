#pragma once

#include "TextureFormat.h"
#include "FboAttachmentType.h"

struct CreateFboAttachmentDTO
{
    unsigned int width;
    unsigned int height;
    FboAttachmentType type;
    TextureFormat captureFormat;
    TextureFormat internalFormat;
};