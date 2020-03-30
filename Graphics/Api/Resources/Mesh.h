#pragma once

#include <vector>
#include <string>
#include <glm/mat4x4.hpp>


struct Mesh
{
    struct __declspec(dllexport) Vertex
    {
        float x;
        float y;
        float z;
        float uv_x;
        float uv_y;
        float normal_x;
        float normal_y;
        float normal_z;
        float tangent_x;
        float tangent_y;
        float tangent_z;
        float bitagent_x;
        float bitagent_y;
        float bitagent_z;
        float bone_weights[4];
        float bone_index[4];
    };

    struct __declspec(dllexport) Polygon
    {
        unsigned int a = 0;
        unsigned int b = 0;
        unsigned int c = 0;
    };

    struct Bone
    {
        glm::mat4 bindPose;
    };

    std::string externalMeshIdentifier;
    std::vector<Vertex> vertices;
    std::vector<Polygon> indices;
    glm::mat4 bindPoseMatrix;
};
