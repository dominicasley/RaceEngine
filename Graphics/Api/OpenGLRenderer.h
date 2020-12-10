#pragma once

#include <spdlog/logger.h>
#include <map>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <tiny_gltf.h>
#include "../Services/SceneService.h"

typedef std::pair<unsigned int, std::string> UniformKey;
typedef std::map<UniformKey, unsigned int> UniformPool;

struct Scene;

class Material;

struct Mesh;

class Shader;

class Texture;

class OpenGLRenderer
{
private:
    UniformPool uniformPool;
    spdlog::logger& logger;
    RenderableEntityService& renderableEntityService;
    SceneService& sceneService;
    CameraService& cameraService;

private:
    std::vector<unsigned int>& bindMesh(std::vector<unsigned int>& vbos,
                                   tinygltf::Model* model, tinygltf::Mesh& mesh);

    void bindModelNodes(std::vector<unsigned int>& vbos, tinygltf::Model* model, tinygltf::Node& node);

public:
    explicit OpenGLRenderer(
        spdlog::logger& logger,
        SceneService& sceneService,
        RenderableEntityService& renderableEntityService,
        CameraService& cameraService);

    bool init();
    void draw(const Scene&);
    void drawMesh(tinygltf::Model* model, tinygltf::Mesh& mesh);
    void drawModelNodes(RenderableEntity* entity, tinygltf::Model* model, tinygltf::Node& node);
    void bindMaterial(const Material*);
    void upload(tinygltf::Model* model);
    unsigned int createShaderObject(const Shader&);
    unsigned int createTexture(const tinygltf::Texture& texture, tinygltf::Image& image);
    void setViewport(int width, int height);

    // OGL SPECIFIC
    bool compileShader(unsigned int id, const std::string& source);

    unsigned int getUniformLocation(unsigned int, const std::string&);

    void setProgramUniform(unsigned int, const std::string&, int);

    void setProgramUniform(unsigned int, const std::string&, float data);

    void setProgramUniform(unsigned int, const std::string&, double data);

    void setProgramUniform(unsigned int, const std::string&, const glm::vec2& data);

    void setProgramUniform(unsigned int, const std::string&, const glm::vec3& data);

    void setProgramUniform(unsigned int, const std::string&, const glm::vec4& data);

    void setProgramUniform(unsigned int, const std::string&, const glm::mat3& data);

    void setProgramUniform(unsigned int, const std::string&, const glm::mat4& data);

    void setProgramUniform(unsigned int, const std::string&, const std::vector<glm::vec2>& data);

    void setProgramUniform(unsigned int, const std::string&, const std::vector<glm::vec3>& data);

    void setProgramUniform(unsigned int, const std::string&, const std::vector<glm::vec4>& data);

    void setProgramUniform(unsigned int, const std::string&, const std::vector<glm::mat3>& data);

    void setProgramUniform(unsigned int, const std::string&, const std::vector<glm::mat4>& data);
};

