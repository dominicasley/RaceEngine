#pragma once

#include <spdlog/logger.h>
#include <map>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <Shared/Services/MemoryStorageService.h>
#include "../Models/Scene/Fbo.h"
#include "../Models/Scene/ShaderDescriptor.h"
#include "../Models/Scene/PixelDataType.h"
#include "../Models/Scene/TextureFormat.h"
#include "../Services/SceneManagerService.h"
#include "../Services/RenderableEntityService.h"

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
    MemoryStorageService& memoryStorageService;
    RenderableEntityService& renderableEntityService;
    SceneManagerService& sceneManagerService;
    std::optional<Resource<Material>> currentlyBoundMaterial;
    unsigned int quadVao;
    std::vector<float> vertices;
    int viewportWidth;
    int viewportHeight;

public:
    explicit OpenGLRenderer(
        spdlog::logger& logger,
        RenderableEntityService& renderableEntityService,
        SceneManagerService& sceneManagerService,
        MemoryStorageService& memoryStorageService);

    bool init();
    void draw(Scene& scene, Camera& camera, float delta);
    void drawFullScreenQuad(const Resource<Shader>& shader, const std::vector<Resource<FboAttachment>>& textures) const;
    void drawFullScreenQuad(const Resource<Shader>& shader, const Resource<FboAttachment>& attachment) const;
    void bindMaterial(const Resource<Material>& material);
    void upload(const Resource<Model>& model);
    void setViewport(int width, int height);
    std::optional<unsigned int> createShaderObject(const ShaderDescriptor& shaderDescriptor);
    [[nodiscard]] unsigned int createTexture(const Texture& texture) const;
    [[nodiscard]] unsigned int createCubeMap(const Texture& front, const Texture& back, const Texture& left, const Texture& right, const Texture& top, const Texture& bottom) const;
    [[nodiscard]] unsigned int createFbo(const Fbo& fbo) const;
    void deleteFbo(Fbo& fbo) const;
    [[nodiscard]] unsigned int getTextureDataType(PixelDataType texture) const;
    [[nodiscard]] unsigned int getTextureFormat(TextureFormat texture) const;
    [[nodiscard]] unsigned int getInternalFormatFromBitsPerPixel(int bitsPerPixel) const;

private:
    void createQuad();
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

