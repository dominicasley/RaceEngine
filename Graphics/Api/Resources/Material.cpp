#include "Material.h"


Material::Material()
{
}

void Material::setTexture(const unsigned int index, const unsigned int textureId)
{
    if (index < maxTextureAllocations)
    {
        this->textures[index] = textureId;
    }
}

void Material::setShader(const unsigned int shaderId)
{
    this->shaderId = shaderId;
}

unsigned int Material::getTexture(const unsigned int &index) const
{
    if (index < maxTextureAllocations)
    {
        return this->textures[index];
    }

    return 0;
}

const std::array<unsigned int, Material::maxTextureAllocations> &Material::getTextures() const
{
    return this->textures;
}

unsigned int Material::getShader() const
{
    return this->shaderId;
}

