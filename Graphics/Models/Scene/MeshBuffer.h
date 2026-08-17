#pragma once

#include <cstddef>
#include <optional>
#include <vector>

struct MeshBuffer {
    int target;
    std::optional<unsigned int> gpuId;
    size_t length;
    size_t offset;
    size_t stride;
    std::vector<unsigned char> data;
};
