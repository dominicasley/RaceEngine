#pragma once

#include <spdlog/logger.h>
#include <map>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../Services/SceneService.h"
#include "../Services/SceneManagerService.h"

typedef std::pair<unsigned int, std::string> UniformKey;
typedef std::map<UniformKey, unsigned int> UniformPool;

struct Scene;
struct Material;
struct Mesh;
struct Model;
struct Shader;
struct Texture;

class OpenGLRenderer
{
private:
    UniformPool uniformPool;
    spdlog::logger& logger;
    RenderableEntityService& renderableEntityService;
    SceneService& sceneService;
    SceneManagerService& sceneManagerService;
    CameraService& cameraService;

private:
    std::vector<unsigned int>& bindMesh(std::vector<unsigned int>& vbos, const Mesh& mesh);

public:
    explicit OpenGLRenderer(
        spdlog::logger& logger,
        RenderableEntityService& renderableEntityService,
        SceneService& sceneService,
        SceneManagerService& sceneManagerService,
        CameraService& cameraService);

    bool init();
    void draw(const Scene&, float delta);
    void drawMesh(const RenderableMesh& mesh);
    void bindMaterial(const Material* material);
    void upload(Model* model);
    unsigned int createShaderObject(const Shader&);
    unsigned int createTexture(Texture* texture) const;
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

