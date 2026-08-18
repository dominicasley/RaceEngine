module;

#include <optional>
#include <string>
#include <vector>

export module raceengine.graphics.models:Texture;

import raceengine.resource;

namespace raceengine
{

// DepthComponent leaves the depth precision to the driver — GL's unsized GL_DEPTH_COMPONENT is
// usually 24-bit fixed point, where Vulkan has no unsized formats and picks D32_SFLOAT. That is
// harmless for a depth *test* and not harmless for a depth map a shader compares against: the same
// bias would clear the quantisation on one backend and not the other. DepthComponent32F names the
// precision, so both backends store the same bits and one bias serves both.
export enum class TextureFormat { R, RG, RGB, RGBA, RGBA16F, RGBA32F, DepthComponent, DepthComponent32F, Unknown };

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
