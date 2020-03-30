#include "Texture.h"
#include <utility>

Texture::Texture() : bitsPerPixel(0), width(0), height(0)
{

}

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
    return Texture();
}

