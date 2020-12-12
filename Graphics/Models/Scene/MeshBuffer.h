#pragma once

struct MeshBuffer {
    int target;
    size_t length;
    size_t offset;
    std::vector<unsigned char> data;
};