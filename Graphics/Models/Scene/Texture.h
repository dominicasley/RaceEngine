#pragma once

#include <optional>
#include <string>
#include <vector>
#include "TextureFormat.h"
#include "PixelDataType.h"

struct Texture
{
    std::string name;
    std::optional<unsigned int> gpuResourceId;
    TextureFormat format;
    PixelDataType pixelDataType;
    unsigned int width;
    unsigned int height;
    unsigned int bitsPerPixel;
    std::vector<unsigned char> data;
};