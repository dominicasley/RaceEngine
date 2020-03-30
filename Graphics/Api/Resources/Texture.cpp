#include "Texture.h"
#include <utility>
#include <FreeImage.h>

Texture::Texture(
        std::vector<unsigned char> data,
        const unsigned int width,
        const unsigned int height,
        const unsigned int bitsPerPixel) :
        data(std::move(data)),
        width(width),
        height(height),
        bitsPerPixel(bitsPerPixel)
{
}

Texture Texture::fromFile(const std::string &fileName)
{
    auto format = FIF_UNKNOWN;
    format = FreeImage_GetFileType(fileName.c_str(), 0);

    if (format == FIF_UNKNOWN)
    {
        format = FreeImage_GetFIFFromFilename(fileName.c_str());
    }

    if (!FreeImage_FIFSupportsReading(format))
    {
        return Texture(std::vector<unsigned char>(), 0, 0, 0);
    }

    const auto image = FreeImage_Load(format, fileName.c_str());

    if (!image)
    {
        return Texture(std::vector<unsigned char>(), 0, 0, 0);
    }

    const auto imageData = FreeImage_GetBits(image);

    auto texture = Texture(
            std::vector<unsigned char>(imageData, imageData + FreeImage_GetMemorySize(image)),
            FreeImage_GetWidth(image),
            FreeImage_GetHeight(image),
            FreeImage_GetBPP(image));

    FreeImage_Unload(image);

    return texture;
}
