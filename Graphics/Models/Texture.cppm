module;

#include <optional>
#include <string>
#include <vector>

export module raceengine.graphics.models:Texture;

import raceengine.resource;

namespace raceengine
{

export enum class TextureFormat { R, RG, RGB, RGBA, RGBA16F, RGBA32F, DepthComponent, Unknown };

export enum class PixelDataType { UnsignedShort, UnsignedByte, Float };

export struct Texture
{
    std::string name;
    std::optional<unsigned int> gpuResourceId{};
    TextureFormat format;
    PixelDataType pixelDataType;
    unsigned int width;
    unsigned int height;
    unsigned int bitsPerPixel{};
    std::vector<unsigned char> data;
};

export struct CubeMap
{
    unsigned int gpuResourceId;
    Resource<Texture> front;
    Resource<Texture> back;
    Resource<Texture> left;
    Resource<Texture> right;
    Resource<Texture> top;
    Resource<Texture> bottom;
};

} // namespace raceengine
