#pragma once

#include "Texture.h"

struct CubeMap {
    unsigned int gpuResourceId;
    Texture* front;
    Texture* back;
    Texture* left;
    Texture* right;
    Texture* top;
    Texture* bottom;
};