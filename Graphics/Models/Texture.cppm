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

// Whether the bytes are an sRGB *encoding* of colour or a measurement stored directly. Nothing
// about the pixels says which: base colour and emissive are authored through a display, normals
// and roughness and occlusion are numbers, and both arrive as the same eight-bit quadruple. Only
// the material slot a texture is bound to knows, so whoever binds it says so here and the backend
// asks the sampler to decode — shading maths is linear and an sRGB byte fed to it raw reads far
// too bright. Linear is the default because it is what a texture with nobody to vouch for it was
// already being treated as.
export enum class ColourSpace { Linear, Srgb };

export struct Texture
{
    std::string name;
    std::optional<unsigned int> gpuResourceId{};
    TextureFormat format;
    PixelDataType pixelDataType;
    ColourSpace colourSpace{ColourSpace::Linear};
    unsigned int width;
    unsigned int height;
    // One for an ordinary image, and the edge of the cube for a colour lookup table — the only
    // thing here that is three-dimensional. It rides on Texture rather than on a model of its own
    // because everything else about it is a sampled image: it is uploaded, bound and released
    // through exactly the same path, and the one place that cares is the image's own creation.
    unsigned int depth = 1;
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
