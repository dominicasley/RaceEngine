#pragma once

#include <vector>
#include <string>

class Texture
{
public:
    const std::vector<unsigned char> data;
    const unsigned int width;
    const unsigned int height;
    const unsigned int bitsPerPixel;

    Texture();
    Texture(std::vector<unsigned char> data, unsigned int width, unsigned int height, unsigned int bitsPerPixel);

    static Texture fromFile(const std::string &fileName);
};
