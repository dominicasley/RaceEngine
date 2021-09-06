#pragma once

#include "Texture.h"

struct CubeMap {
    unsigned int gpuResourceId;
    Resource<Texture> front;
    Resource<Texture> back;
    Resource<Texture> left;
    Resource<Texture> right;
    Resource<Texture> top;
    Resource<Texture> bottom;
};