#pragma once

#include <vector>
#include <memory>

#include "Joint.h"

struct Skeleton
{
public:
    std::map<int, std::unique_ptr<Joint>> joints;
};
