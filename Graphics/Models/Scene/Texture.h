#pragma once

#include <string>
#include <optional>
#include "TextureFormat.h"
#include "PixelDataType.h"

struct Texture {
    std::string name;
    std::optional<unsigned int> gpuResourceId;
    TextureFormat format;
    PixelDataType pixelDataType;
    int width;
    int height;
    std::vector<unsigned char> data;
};