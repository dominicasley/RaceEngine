#pragma once

#include "Texture.h"
#include "Shader.h"
#include <array>
#include <map>

#include "Uniform.h"

class Material
{
    constexpr static unsigned int maxTextureAllocations = 8;

private:
    std::array<unsigned int, maxTextureAllocations> textures;
    unsigned int shaderId;

public:
    Material();

    void setTexture(unsigned int index, unsigned int textureId);

    void setShader(unsigned int shaderId);

    const std::array<unsigned int, maxTextureAllocations> &getTextures() const;

    unsigned int getTexture(const unsigned int &index) const;

    unsigned int getShader() const;
};
