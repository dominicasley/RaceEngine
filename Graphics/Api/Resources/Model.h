#pragma once

#include <map>
#include <memory>
#include "Mesh.h"
#include "Skeleton.h"
#include "Animation.h"

class Model
{
public:
    std::map<std::string, Mesh> meshes;
    std::shared_ptr<Skeleton> skeleton;
    std::map<std::string, Animation> animations;

    Model();

    static Model fromFile(const std::string &fileName);

    static Model fromFile(const std::string &fileName, const std::vector<std::string> &animationFiles);
};
