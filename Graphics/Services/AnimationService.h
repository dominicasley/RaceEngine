#pragma once

#include <spdlog/logger.h>
#include <tiny_gltf.h>
#include "../../Io/Utility/AccessorUtility.h"
#include "../Models/Scene/Skeleton.h"

class AnimationService
{
public:
    explicit AnimationService(spdlog::logger& logger);
};


