#pragma once

#include <spdlog/logger.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
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

struct UniformKeyHash
{
    using is_transparent = void;

    [[nodiscard]] size_t operator()(const std::pair<unsigned int, std::string_view>& key) const
    {
        return std::hash<unsigned int>{}(key.first) ^ (std::hash<std::string_view>{}(key.second) << 1u);
    }

    [[nodiscard]] size_t operator()(const std::pair<unsigned int, std::string>& key) const
    {
        return (*this)(std::pair<unsigned int, std::string_view>(key.first, key.second));
    }
};

struct UniformKeyEqual
{
    using is_transparent = void;

    template<typename L, typename R>
    [[nodiscard]] bool operator()(const L& lhs, const R& rhs) const
    {
        return lhs.first == rhs.first && std::string_view(lhs.second) == std::string_view(rhs.second);
    }
};

typedef std::unordered_map<std::pair<unsigned int, std::string>, int, UniformKeyHash, UniformKeyEqual> UniformPool;

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
    float maxAnisotropy = 1.0f;

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
    int getUniformLocation(unsigned int, const char*);
    void setProgramUniform(unsigned int, const char*, int);
    void setProgramUniform(unsigned int, const char*, float data);
    void setProgramUniform(unsigned int, const char*, double data);
    void setProgramUniform(unsigned int, const char*, const glm::vec2& data);
    void setProgramUniform(unsigned int, const char*, const glm::vec3& data);
    void setProgramUniform(unsigned int, const char*, const glm::vec4& data);
    void setProgramUniform(unsigned int, const char*, const glm::mat3& data);
    void setProgramUniform(unsigned int, const char*, const glm::mat4& data);
    void setProgramUniform(unsigned int, const char*, const std::vector<glm::vec2>& data);
    void setProgramUniform(unsigned int, const char*, const std::vector<glm::vec3>& data);
    void setProgramUniform(unsigned int, const char*, const std::vector<glm::vec4>& data);
    void setProgramUniform(unsigned int, const char*, const std::vector<glm::mat3>& data);
    void setProgramUniform(unsigned int, const char*, const std::vector<glm::mat4>& data);
};
