#pragma once

#include <spdlog/logger.h>
#include <map>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../Models/Scene/ShaderDescriptor.h"
#include "../Services/SceneService.h"
#include "../Services/SceneManagerService.h"
#include "../Models/Scene/PixelDataType.h"
#include "../Models/Scene/TextureFormat.h"

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
    Material* currentlyBoundMaterial = nullptr;

    std::vector<unsigned int>& bindMesh(std::vector<unsigned int>& vbos, const Mesh& mesh);

public:
    explicit OpenGLRenderer(
        spdlog::logger& logger,
        RenderableEntityService& renderableEntityService,
        SceneService& sceneService,
        SceneManagerService& sceneManagerService,
        CameraService& cameraService);

    bool init();
    void draw(const Scene* scene, float delta);
    void drawMesh(const RenderableMesh& mesh);
    void drawPrimitives(const std::vector<MeshPrimitive>& primitives) const;
    void bindMaterial(const Material* material);
    void upload(Model* model);
    std::optional<unsigned int> createShaderObject(const ShaderDescriptor& shaderDescriptor);
    unsigned int createTexture(Texture* texture) const;
    unsigned int createCubeMap(Texture* front, Texture* back, Texture* left, Texture* right, Texture* top, Texture* bottom) const;
    void setViewport(int width, int height);
    [[nodiscard]] unsigned int getTextureDataType(PixelDataType texture) const;
    [[nodiscard]] unsigned int getTextureFormat(TextureFormat texture) const;
    [[nodiscard]] unsigned int getInternalFormatFromBitsPerPixel(int bitsPerPixel) const;

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

