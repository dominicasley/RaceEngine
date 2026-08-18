module;

// glad must precede GLFW: its include guard stops glfw3.h dragging in the system GL headers.
#include <glad/gl.h>

#include <GLFW/glfw3.h>

#include <cstring>
#include <expected>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <spdlog/logger.h>
#include <stb_image_write.h>

// An implementation partition, not an interface one: naming it `export module` put the whole
// backend — and glad and GLFW with it — into the interface closure of raceengine.graphics, and
// so into every translation unit that imports raceengine. Only :RenderBackendFactoryImpl imports
// this, and nothing outside this module can.
module raceengine.graphics:OpenGLRenderer;

import :FrameDiagnostics;
import :GraphicsApi;
import :IRenderBackend;
import :RenderContract;
import :SceneManagerService;
import :RenderableEntityService;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

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

    template <typename L, typename R> [[nodiscard]] bool operator()(const L& lhs, const R& rhs) const
    {
        return lhs.first == rhs.first && std::string_view(lhs.second) == std::string_view(rhs.second);
    }
};

typedef std::unordered_map<std::pair<unsigned int, std::string>, int, UniformKeyHash, UniformKeyEqual> UniformPool;

class OpenGLRenderer : public IRenderBackend
{
private:
    UniformPool uniformPool;
    spdlog::logger& logger;
    // Every skipped draw, view or pass is counted here instead of at a per-site boolean; see
    // FrameDiagnostics for why the sites do not throttle themselves any more.
    FrameDiagnostics& diagnostics;
    MemoryStorageService& memoryStorageService;
    RenderableEntityService& renderableEntityService;
    SceneManagerService& sceneManagerService;
    std::optional<Resource<Material>> currentlyBoundMaterial;
    std::optional<unsigned int> sceneEnvironmentGpuId;
    unsigned int quadVao;
    std::vector<float> vertices;
    int viewportWidth;
    int viewportHeight;
    float maxAnisotropy = 1.0f;
    // Owned GL object ids, appended at the creation sites and released in the destructor.
    // The destructor runs while the GLFW context is still current (see Engine member order).
    std::vector<unsigned int> createdPrograms;
    std::vector<unsigned int> createdVaos;
    std::vector<unsigned int> createdVbos;
    std::vector<unsigned int> createdTextures;
    std::vector<unsigned int> createdFbos;
    // Whether this frame recorded a present pass. GL has no swapchain to acquire, so the
    // frame is bookkeeping only — but the fallback it drives is real: a frame with no
    // presenter leaves the window showing the clear colour rather than whatever the driver
    // left in the back buffer, which is what Vulkan's clear-only pass already guaranteed.
    bool presentRecorded = false;

public:
    explicit OpenGLRenderer(spdlog::logger& logger, FrameDiagnostics& diagnostics,
                            RenderableEntityService& renderableEntityService, SceneManagerService& sceneManagerService,
                            MemoryStorageService& memoryStorageService);
    ~OpenGLRenderer() override;

    [[nodiscard]] std::expected<void, std::string> init() override;
    void setViewport(int width, int height) override;

    [[nodiscard]] bool beginFrame() override;
    void recordView(Scene& scene, Camera& camera, float delta) override;
    void recordPresent(const Resource<Shader>& shader, const Resource<FboAttachment>& attachment) override;
    void endFrame() override;

    [[nodiscard]] std::expected<unsigned int, std::string>
    createShaderObject(const ShaderDescriptor& shaderDescriptor) override;
    [[nodiscard]] std::expected<unsigned int, std::string> createCubeMap(const Texture& front, const Texture& back,
                                                                         const Texture& left, const Texture& right,
                                                                         const Texture& top,
                                                                         const Texture& bottom) override;
    [[nodiscard]] std::expected<unsigned int, std::string> createFbo(const Fbo& fbo) override;
    void deleteFbo(Fbo& fbo) override;
    void releaseGpuResource(GpuResourceKind kind, unsigned int gpuResourceId) override;
    void releaseMaterial(const Resource<Material>& material) override;
    [[nodiscard]] GpuResourceCensus gpuResourceCensus() const override;

    [[nodiscard]] std::expected<void, std::string> captureFrame(const std::string& path) override;

private:
    // False means the pass was not recorded because its shader handle names nothing; the caller
    // knows which pass it was asking for and counts it.
    [[nodiscard]] bool drawFullScreenQuad(const Resource<Shader>& shader,
                                          const std::vector<Resource<FboAttachment>>& textures) const;
    void bindMaterial(const Material& material, unsigned int programId);
    void uploadLights(unsigned int programId, const Scene& scene);
    void upload(const Resource<Model>& model);
    void uploadMaterialTextures(const Resource<Material>& material);
    void uploadTexture(const Resource<Texture>& texture);
    [[nodiscard]] unsigned int createTexture(const Texture& texture);
    [[nodiscard]] unsigned int getTextureDataType(PixelDataType texture) const;
    [[nodiscard]] unsigned int getTextureFormat(TextureFormat texture) const;
    [[nodiscard]] unsigned int getInternalFormatFromBitsPerPixel(int bitsPerPixel) const;
    void createQuad();
    // Reports the compiler's own words rather than logging them: the only caller already has an
    // error channel, and a stage that would not compile is the reason the program will not link.
    [[nodiscard]] std::expected<void, std::string> compileShader(unsigned int id, const std::string& source);
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
    void setProgramUniform(unsigned int, const char*, const std::vector<glm::mat4>& data, size_t count);
};

namespace
{

struct LightUniformNames
{
    std::string position;
    std::string diffuse;
    std::string specular;
    std::string ambient;
    std::string attenuation;
};

// GL addresses a struct-array uniform by name, so the names are built once and the draw path
// pays only the uniform-pool lookup.
const std::vector<LightUniformNames> lightUniformNames = []
{
    std::vector<LightUniformNames> names;
    names.reserve(maxLights);

    for (auto index = 0u; index < maxLights; index++)
    {
        const auto prefix = "lights[" + std::to_string(index) + "].";
        names.push_back(LightUniformNames{.position = prefix + "position",
                                          .diffuse = prefix + "diffuse",
                                          .specular = prefix + "specular",
                                          .ambient = prefix + "ambient",
                                          .attenuation = prefix + "attenuation"});
    }

    return names;
}();

[[nodiscard]] GLenum textureUnit(const MaterialTextureSlot slot)
{
    return static_cast<GLenum>(GL_TEXTURE0) + textureBinding(slot, GraphicsApi::OpenGL);
}

void GLAD_API_PTR glDebugMessageHandler(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei,
                                        const GLchar* message, const void* userParam)
{
    auto& logger = *static_cast<spdlog::logger*>(const_cast<void*>(userParam));

    switch (severity)
    {
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        logger.debug("GL debug message [source 0x{:x}, type 0x{:x}, id {}]: {}", source, type, id, message);
        break;
    case GL_DEBUG_SEVERITY_LOW:
    case GL_DEBUG_SEVERITY_MEDIUM:
        logger.warn("GL debug message [source 0x{:x}, type 0x{:x}, id {}]: {}", source, type, id, message);
        break;
    default:
        logger.error("GL debug message [source 0x{:x}, type 0x{:x}, id {}]: {}", source, type, id, message);
        break;
    }
}
} // namespace

OpenGLRenderer::OpenGLRenderer(spdlog::logger& logger, FrameDiagnostics& diagnostics,
                               RenderableEntityService& renderableEntityService,
                               SceneManagerService& sceneManagerService, MemoryStorageService& memoryStorageService) :
    logger(logger),
    diagnostics(diagnostics),
    memoryStorageService(memoryStorageService),
    renderableEntityService(renderableEntityService),
    sceneManagerService(sceneManagerService)
{
}

OpenGLRenderer::~OpenGLRenderer()
{
    for (const auto programId : createdPrograms)
    {
        glDeleteProgram(programId);
    }

    if (!createdVaos.empty())
    {
        glDeleteVertexArrays(static_cast<GLsizei>(createdVaos.size()), createdVaos.data());
    }

    if (!createdVbos.empty())
    {
        glDeleteBuffers(static_cast<GLsizei>(createdVbos.size()), createdVbos.data());
    }

    if (!createdTextures.empty())
    {
        glDeleteTextures(static_cast<GLsizei>(createdTextures.size()), createdTextures.data());
    }

    if (!createdFbos.empty())
    {
        glDeleteFramebuffers(static_cast<GLsizei>(createdFbos.size()), createdFbos.data());
    }
}

std::expected<void, std::string> OpenGLRenderer::init()
{
    if (gladLoadGL(glfwGetProcAddress) == 0)
    {
        return std::unexpected("glad could not load the OpenGL entry points from the GLFW context");
    }

    if (glDebugMessageCallback != nullptr)
    {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(glDebugMessageHandler, &logger);
    }

    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &maxAnisotropy);

    glClearColor(clearColour[0], clearColour[1], clearColour[2], clearColour[3]);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glEnable(GL_CULL_FACE);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

    glCullFace(GL_BACK);

    createQuad();

    return {};
}

// GL has no image to acquire and no command buffer to open: the context is always ready, so
// the frame exists to say when it started and to let endFrame see whether anything presented.
bool OpenGLRenderer::beginFrame()
{
    presentRecorded = false;
    return true;
}

// Nothing to submit — GL commands went to the driver as they were issued — and nothing to
// present either: the buffer swap belongs to the window, which owns the surface. What is
// left is the frame's guarantee that something reached the back buffer.
void OpenGLRenderer::endFrame()
{
    if (presentRecorded)
    {
        return;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, viewportWidth, viewportHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

// Handles are indices now, so `resource->field` would be a storage lookup — and a mutex — per
// field. Every resolution in this pass is therefore hoisted to the outermost loop where its
// handle is constant: the model and the instance shader once per entity, the mesh once per mesh,
// the material once per primitive because that is where it varies. The pass costs
// O(entities + meshes + primitives) lookups instead of the ~15 pointer chases per primitive it
// used to make, and each lookup is one compare against the slot's generation.
//
// A borrow taken here stays valid for the rest of the pass: removal is a main-thread operation
// between frames, and upload() writes through mutate(), which writes the element in place rather
// than assigning a new one over it.
void OpenGLRenderer::recordView(Scene& scene, Camera& camera, float delta)
{
    currentlyBoundMaterial.reset();

    sceneEnvironmentGpuId.reset();
    if (scene.environment.has_value())
    {
        if (const auto* environment = memoryStorageService.cubeMaps.find(scene.environment.value());
            environment != nullptr)
        {
            sceneEnvironmentGpuId = environment->gpuResourceId;
        }
    }

    // The pass states its own target and its own state instead of inheriting either. A view
    // is one of N recorded into the same frame, so what the previous view left bound — the
    // default framebuffer, or culling switched off by its last non-opaque material — says
    // nothing about what this one needs.
    auto target = 0u;
    if (camera.output.has_value())
    {
        const auto* output = memoryStorageService.frameBuffers.find(camera.output.value());

        if (output != nullptr && output->gpuResourceId.has_value())
        {
            target = output->gpuResourceId.value();
        }
        else
        {
            diagnostics.record(FrameDiagnostic::MissingCameraOutput,
                               [] { return std::string("drawing to the default framebuffer instead"); });
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, target);
    glViewport(0, 0, viewportWidth, viewportHeight);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for (auto& entity : scene.models)
    {
        // One resolution per entity for the whole nest below. A stale handle here — the model was
        // unloaded while its renderable is still in the scene — is what an entity with nothing
        // left to draw looks like, and skipping it is the only correct answer: the slot it names
        // may already hold a different model.
        const auto* model = memoryStorageService.models.find(entity.model);
        const auto* instanceShader = memoryStorageService.shaders.find(entity.shader);
        if (model == nullptr || instanceShader == nullptr)
        {
            diagnostics.record(FrameDiagnostic::StaleModelHandle,
                               [&] { return "model slot " + std::to_string(entity.model.index); });

            continue;
        }

        // The instance's shader, not the material's: materials live in shared storage, so two
        // renderables built from one model would otherwise restyle each other.
        const auto programId = instanceShader->gpuResourceId;

        if (entity.beforeDraw.has_value())
        {
            (*entity.beforeDraw)();
        }

        for (auto& renderableMesh : entity.meshes)
        {
            const auto* mesh = memoryStorageService.meshes.find(renderableMesh.mesh);
            if (mesh == nullptr)
            {
                diagnostics.record(FrameDiagnostic::StaleMeshHandle,
                                   [&] { return "mesh slot " + std::to_string(renderableMesh.mesh.index); });

                continue;
            }

            if (!mesh->gpuResourceId.has_value())
            {
                // upload() writes through mutate(), which writes the element in place — so this
                // borrow still names it, now carrying the ids the upload produced. Under the old
                // update() the element was copy-assigned and everything inside it was freed, and
                // this loop was safe only because it re-read through the handle afterwards.
                upload(entity.model);
            }

            if (!mesh->gpuResourceId.has_value())
            {
                diagnostics.record(FrameDiagnostic::MeshNotUploaded, [&] { return "mesh " + mesh->name; });

                continue;
            }

            auto entityModelMatrix = sceneManagerService.modelMatrix(entity.node) * mesh->modelMatrix;

            // Bound by reference into the mesh's palette buffer: copying it here would undo the
            // per-frame allocation the service avoids. The clamp limits the upload count instead.
            const auto& joints = renderableEntityService.joints(renderableMesh, delta);
            const auto jointCount = std::min(joints.size(), static_cast<size_t>(maxJoints));

            if (joints.size() > maxJoints)
            {
                diagnostics.record(FrameDiagnostic::JointLimitExceeded,
                                   [&]
                                   {
                                       return "the GL joint uniform array holds " + std::to_string(maxJoints) +
                                              " of the " + std::to_string(joints.size()) + " supplied for mesh " +
                                              mesh->name;
                                   });
            }

            for (const auto& primitive : mesh->meshPrimitives)
            {
                if (!model->meshBuffers[static_cast<size_t>(primitive.meshBufferIndex)].gpuId.has_value())
                {
                    diagnostics.record(FrameDiagnostic::MeshBufferNotUploaded, [&] { return "mesh " + mesh->name; });

                    continue;
                }

                const auto* material = primitive.material.has_value()
                                           ? memoryStorageService.materials.find(primitive.material.value())
                                           : nullptr;

                if (material == nullptr || !material->shader.has_value())
                {
                    diagnostics.record(FrameDiagnostic::PrimitiveWithoutMaterial, [&] { return "mesh " + mesh->name; });

                    continue;
                }

                setProgramUniform(programId, "localToScreen4x4Matrix",
                                  camera.modelViewProjectionMatrix * entityModelMatrix);
                setProgramUniform(programId, "localToView4x4Matrix", camera.modelViewMatrix * entityModelMatrix);
                setProgramUniform(programId, "localToWorld4x4Matrix", entityModelMatrix);

                setProgramUniform(programId, "normalMatrix",
                                  glm::transpose(glm::inverse(glm::mat3(camera.modelViewMatrix * entityModelMatrix))));
                setProgramUniform(programId, "jointTransformationMatrixes", joints, jointCount);
                setProgramUniform(programId, "animated", !joints.empty());

                setProgramUniform(programId, "textureTransform", material->transform);

                if (!currentlyBoundMaterial.has_value() || currentlyBoundMaterial.value() != primitive.material.value())
                {
                    bindMaterial(*material, programId);

                    if (!material->opaque)
                    {
                        glDisable(GL_CULL_FACE);
                    }
                    else
                    {
                        glEnable(GL_CULL_FACE);
                    }

                    setProgramUniform(programId, "u_baseColour", material->baseColour);
                    setProgramUniform(programId, "u_metalness", material->metalness);
                    setProgramUniform(programId, "u_roughness", material->roughness);

                    currentlyBoundMaterial = primitive.material.value();

                    setProgramUniform(programId, "cameraPosition", camera.position);
                    setProgramUniform(programId, "modelView3x3Matrix", glm::mat3(camera.modelViewMatrix));

                    uploadLights(programId, scene);
                }

                if (!primitive.gpuVao.has_value())
                {
                    diagnostics.record(FrameDiagnostic::PrimitiveNotUploaded, [&] { return "mesh " + mesh->name; });

                    continue;
                }

                glBindVertexArray(primitive.gpuVao.value());

                glDrawElements(primitive.mode, static_cast<GLsizei>(primitive.elementCount), primitive.componentType,
                               reinterpret_cast<const void*>(primitive.byteOffset));
            }
        }

        glBindVertexArray(0);

        if (entity.afterDraw.has_value())
        {
            (*entity.afterDraw)();
        }
    }

    // Each link of the chain binds the buffer it writes; a post-process with no output of its
    // own writes the default framebuffer, which is what it did when it inherited the binding
    // the scene pass left behind.
    for (const auto& postProcessKey : camera.postProcesses)
    {
        const auto* postProcess = memoryStorageService.postProcesses.find(postProcessKey);
        if (postProcess == nullptr)
        {
            diagnostics.record(FrameDiagnostic::PostProcessSkipped,
                               [&] { return "post-process slot " + std::to_string(postProcessKey.index); });

            continue;
        }

        auto postProcessTarget = 0u;
        if (postProcess->output.has_value())
        {
            const auto* output = memoryStorageService.frameBuffers.find(postProcess->output.value());
            if (output != nullptr && output->gpuResourceId.has_value())
            {
                postProcessTarget = output->gpuResourceId.value();
            }
        }

        glBindFramebuffer(GL_FRAMEBUFFER, postProcessTarget);

        if (!drawFullScreenQuad(postProcess->shader, postProcess->inputs))
        {
            diagnostics.record(FrameDiagnostic::PostProcessSkipped,
                               [&] { return "shader slot " + std::to_string(postProcess->shader.index); });
        }
    }
}

// The frame's last pass. It binds the default framebuffer itself rather than relying on a
// view having left it bound, which is what made the presenter depend on a scene pass running
// first.
void OpenGLRenderer::recordPresent(const Resource<Shader>& shaderKey, const Resource<FboAttachment>& attachmentKey)
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (!drawFullScreenQuad(shaderKey, std::vector<Resource<FboAttachment>>{attachmentKey}))
    {
        diagnostics.record(FrameDiagnostic::PresentPassSkipped,
                           [&] { return "shader slot " + std::to_string(shaderKey.index); });

        return;
    }

    presentRecorded = true;
}

// A scene with no lights uploads lightCount 0: both shader loops then contribute nothing and
// the ambient floor is zero, leaving the image-based term as the only lighting. That is a
// legitimate scene, not a failure, so nothing is logged.
void OpenGLRenderer::uploadLights(const unsigned int programId, const Scene& scene)
{
    auto uploaded = 0u;
    auto declared = 0u;

    for (const auto& light : scene.lights)
    {
        declared++;

        if (uploaded >= maxLights)
        {
            continue;
        }

        const auto& names = lightUniformNames[uploaded];
        setProgramUniform(programId, names.position.c_str(), light.position);
        setProgramUniform(programId, names.diffuse.c_str(), light.diffuse);
        setProgramUniform(programId, names.specular.c_str(), light.specular);
        setProgramUniform(programId, names.ambient.c_str(), light.ambient);
        setProgramUniform(programId, names.attenuation.c_str(), light.attenuation);
        uploaded++;
    }

    if (declared > maxLights)
    {
        diagnostics.record(FrameDiagnostic::LightLimitExceeded,
                           [&]
                           {
                               return "the GL light uniform array holds " + std::to_string(maxLights) + " of the " +
                                      std::to_string(declared) + " declared";
                           });
    }

    setProgramUniform(programId, "lightCount", static_cast<int>(uploaded));
}

void OpenGLRenderer::bindMaterial(const Material& material, const unsigned int programId)
{
    glUseProgram(programId);

    // One resolution per slot, and a slot whose handle has gone stale reads exactly like a slot
    // the material never had: nothing bound, and the shader's use-flag off. The behaviour for a
    // slot present but not yet uploaded is unchanged — the unit keeps whatever was there, which
    // is what the missing else branch has always meant here.
    const auto bindSlot =
        [&](const std::optional<Resource<Texture>>& textureKey, const MaterialTextureSlot slot, const char* useUniform)
    {
        const auto* texture = textureKey.has_value() ? memoryStorageService.textures.find(textureKey.value()) : nullptr;

        if (texture == nullptr)
        {
            glActiveTexture(textureUnit(slot));
            glBindTexture(GL_TEXTURE_2D, 0);
            setProgramUniform(programId, useUniform, false);

            return;
        }

        if (texture->gpuResourceId.has_value())
        {
            glActiveTexture(textureUnit(slot));
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(texture->gpuResourceId.value()));
            setProgramUniform(programId, useUniform, true);
        }
    };

    bindSlot(material.albedo, MaterialTextureSlot::Diffuse, "u_useDiffuseTexture");
    bindSlot(material.normal, MaterialTextureSlot::Normal, "u_useNormalTexture");
    bindSlot(material.metallicRoughness, MaterialTextureSlot::Specular, "u_useSpecularTexture");
    bindSlot(material.emissive, MaterialTextureSlot::Emissive, "u_useEmissiveTexture");
    bindSlot(material.occlusion, MaterialTextureSlot::Occlusion, "u_useOcclusionTexture");

    const auto* materialEnvironment =
        material.environment.has_value() ? memoryStorageService.cubeMaps.find(material.environment.value()) : nullptr;

    glActiveTexture(textureUnit(MaterialTextureSlot::Environment));
    if (materialEnvironment != nullptr)
    {
        glBindTexture(GL_TEXTURE_CUBE_MAP, materialEnvironment->gpuResourceId);
    }
    else if (sceneEnvironmentGpuId.has_value())
    {
        glBindTexture(GL_TEXTURE_CUBE_MAP, sceneEnvironmentGpuId.value());
    }
    else
    {
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    }

    for (auto i = 0u; i < material.textures.size(); i++)
    {
        const auto* texture = memoryStorageService.textures.find(material.textures[i]);

        if (texture != nullptr && texture->gpuResourceId.has_value())
        {
            glActiveTexture(textureUnit(MaterialTextureSlot::Environment) + i + 1);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(texture->gpuResourceId.value()));
        }
    }
}

// The whole upload runs inside one mutate() of the model, with a nested mutate() per mesh. That
// is not incidental: it is the only shape in which the ids can be written to the elements the
// draw path reads without either copying an 18 MB model out and back — which is what
// get()/update() did on the first draw — or const_casting the storage's own reference.
//
// Lock nesting is models -> meshes and models -> materials -> textures, always in that order and
// never back into the same storage, which is the rule mutate() states.
void OpenGLRenderer::upload(const Resource<Model>& modelKey)
{
    memoryStorageService.models.mutate(
        modelKey,
        [&](Model& model)
        {
            for (const auto& meshKey : model.meshes)
            {
                memoryStorageService.meshes.mutate(
                    meshKey,
                    [&](Mesh& mesh)
                    {
                        if (mesh.gpuResourceId.has_value())
                        {
                            return;
                        }

                        for (auto& buffer : model.meshBuffers)
                        {
                            if (buffer.gpuId.has_value())
                            {
                                continue;
                            }

                            GLuint vbo;
                            glGenBuffers(1, &vbo);
                            createdVbos.push_back(vbo);
                            glBindBuffer(static_cast<GLenum>(buffer.target), vbo);
                            glBufferData(static_cast<GLenum>(buffer.target),
                                         static_cast<GLsizeiptr>(buffer.data.size()), buffer.data.data(),
                                         GL_STATIC_DRAW);

                            buffer.gpuId = vbo;
                            buffer.data.clear();
                            buffer.data.shrink_to_fit();
                        }

                        // draw()'s "already uploaded" sentinel. Nothing binds it — each primitive
                        // owns its VAO in gpuVao — but it has to be assigned outside the loop
                        // below: GLTFService drops non-indexed primitives, so meshPrimitives can
                        // be empty, and an unset sentinel makes draw() re-enter upload() on every
                        // frame.
                        auto uploadedSentinel = 0u;

                        for (auto& primitive : mesh.meshPrimitives)
                        {
                            GLuint vao;
                            glGenVertexArrays(1, &vao);
                            glBindVertexArray(vao);
                            createdVaos.push_back(vao);

                            for (const auto& attribute : primitive.attributes)
                            {
                                glBindBuffer(
                                    GL_ARRAY_BUFFER,
                                    model.meshBuffers[static_cast<size_t>(attribute.bufferIndex)].gpuId.value());

                                if (attribute.attributeType.has_value())
                                {
                                    glEnableVertexAttribArray(static_cast<GLuint>(attribute.attributeType.value()));
                                    glVertexAttribPointer(static_cast<GLuint>(attribute.attributeType.value()),
                                                          attribute.size, attribute.componentType,
                                                          attribute.normalized ? GL_TRUE : GL_FALSE, attribute.stride,
                                                          reinterpret_cast<const void*>(attribute.offset));
                                }
                            }

                            // The element buffer binding is VAO state; baking it here removes the
                            // per-draw rebind.
                            glBindBuffer(
                                GL_ELEMENT_ARRAY_BUFFER,
                                static_cast<GLuint>(
                                    model.meshBuffers[static_cast<size_t>(primitive.meshBufferIndex)].gpuId.value()));

                            primitive.gpuVao = vao;
                            uploadedSentinel = vao;

                            glBindVertexArray(0);
                        }

                        mesh.gpuResourceId = uploadedSentinel;
                    });
            }

            for (const auto& materialKey : model.materials)
            {
                uploadMaterialTextures(materialKey);
            }
        });
}

void OpenGLRenderer::uploadMaterialTextures(const Resource<Material>& materialKey)
{
    const auto* material = memoryStorageService.materials.find(materialKey);
    if (material == nullptr)
    {
        return;
    }

    for (const auto& textureKey :
         {material->albedo, material->normal, material->metallicRoughness, material->emissive, material->occlusion})
    {
        if (textureKey.has_value())
        {
            uploadTexture(textureKey.value());
        }
    }

    for (const auto& textureKey : material->textures)
    {
        uploadTexture(textureKey);
    }
}

void OpenGLRenderer::uploadTexture(const Resource<Texture>& textureKey)
{
    // In place: the id is written and the pixels dropped inside the element, so a multi-megabyte
    // payload is never copied through the storage. The const_cast this needed is gone with
    // update() — mutate() is the API that says "write this element", and it is the only one.
    // A stale handle is not an upload that failed, it is a texture that is no longer loaded, so
    // the mutate simply does not run.
    memoryStorageService.textures.mutate(textureKey,
                                         [&](Texture& texture)
                                         {
                                             if (texture.gpuResourceId.has_value())
                                             {
                                                 return;
                                             }

                                             texture.gpuResourceId = createTexture(texture);

                                             // The GPU owns the texels now; shed the CPU copy the
                                             // way the mesh buffers above do.
                                             texture.data.clear();
                                             texture.data.shrink_to_fit();
                                         });
}

std::expected<unsigned int, std::string> OpenGLRenderer::createShaderObject(const ShaderDescriptor& object)
{
    const auto programId = glCreateProgram();
    const auto vertexShaderId = glCreateShader(GL_VERTEX_SHADER);
    const auto fragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);
    const auto tessellationControlShaderId = glCreateShader(GL_TESS_CONTROL_SHADER);
    const auto tessellationEvaluationShaderId = glCreateShader(GL_TESS_EVALUATION_SHADER);
    const auto computeShaderId = glCreateShader(GL_COMPUTE_SHADER);
    const auto geometryShaderId = glCreateShader(GL_GEOMETRY_SHADER);

    // Every GL source is compiled with the contract macros spliced in, so no shader spells a
    // binding index, an attribute location or an array bound the C++ side also holds.
    const auto contract = [](const std::string& source)
    {
        return withShaderContractMacros(source, GraphicsApi::OpenGL);
    };

    // A stage that will not compile is why the link below fails, so its message is carried to
    // the caller with the link log rather than logged here and lost from the report.
    std::string stageDiagnostics;
    const auto compile = [&](const unsigned int shaderId, const std::string& source, const char* stage)
    {
        const auto compiled = compileShader(shaderId, contract(source));
        if (compiled)
        {
            return true;
        }

        stageDiagnostics += (stageDiagnostics.empty() ? "" : "; ") + std::string(stage) + " stage: " + compiled.error();

        return false;
    };

    if (!object.vertexShaderSource.empty())
    {
        if (compile(vertexShaderId, object.vertexShaderSource, "vertex"))
        {
            glAttachShader(programId, vertexShaderId);
        }
    }

    if (!object.fragmentShaderSource.empty())
    {
        if (compile(fragmentShaderId, object.fragmentShaderSource, "fragment"))
        {
            glAttachShader(programId, fragmentShaderId);
        }
    }

    if (!object.tessellationControlShaderSource.empty() && !object.tessellationEvaluationShaderSource.empty())
    {
        const auto control =
            compile(tessellationControlShaderId, object.tessellationControlShaderSource, "tessellation control");
        const auto evaluation = compile(tessellationEvaluationShaderId, object.tessellationEvaluationShaderSource,
                                        "tessellation evaluation");

        if (control && evaluation)
        {
            glAttachShader(programId, tessellationControlShaderId);
            glAttachShader(programId, tessellationEvaluationShaderId);
        }
    }

    if (!object.computeShaderSource.empty())
    {
        if (compile(computeShaderId, object.computeShaderSource, "compute"))
        {
            glAttachShader(programId, computeShaderId);
        }
    }

    if (!object.geometryShaderSource.empty())
    {
        if (compile(geometryShaderId, object.geometryShaderSource, "geometry"))
        {
            glAttachShader(programId, geometryShaderId);
        }
    }

    glLinkProgram(programId);

    auto result = GL_FALSE;
    GLint infoLogLength = 0;
    glGetProgramiv(programId, GL_LINK_STATUS, &result);
    glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &infoLogLength);

    std::string linkLog;
    if (infoLogLength > 1)
    {
        std::vector<char> infoLog(static_cast<size_t>(infoLogLength) + 1);
        glGetProgramInfoLog(programId, infoLogLength, nullptr, infoLog.data());
        linkLog = infoLog.data();

        // A log on a program that linked is a driver remark, not a failure; only the failing
        // case is reported to the caller, so a warning here is the whole story.
        if (result == GL_TRUE)
        {
            logger.warn("shader link log: {}", linkLog);
        }
    }

    glDeleteShader(vertexShaderId);
    glDeleteShader(fragmentShaderId);
    glDeleteShader(tessellationControlShaderId);
    glDeleteShader(tessellationEvaluationShaderId);
    glDeleteShader(computeShaderId);
    glDeleteShader(geometryShaderId);

    if (result != GL_TRUE)
    {
        glDeleteProgram(programId);

        auto reason = std::string("the GL program did not link");
        if (!linkLog.empty())
        {
            reason += ": " + linkLog;
        }
        if (!stageDiagnostics.empty())
        {
            reason += " (" + stageDiagnostics + ")";
        }

        return std::unexpected(reason);
    }

    createdPrograms.push_back(programId);

    return programId;
}

unsigned int OpenGLRenderer::createTexture(const Texture& texture)
{
    unsigned int textureId;

    glGenTextures(1, &textureId);
    createdTextures.push_back(textureId);
    glBindTexture(GL_TEXTURE_2D, textureId);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    auto format = getTextureFormat(texture.format);
    auto type = getTextureDataType(texture.pixelDataType);
    auto internalFormat = static_cast<unsigned int>(GL_RGBA);
    switch (texture.pixelDataType)
    {
    case PixelDataType::Float:
        internalFormat = getInternalFormatFromBitsPerPixel(static_cast<int>(texture.bitsPerPixel));
        break;
    case PixelDataType::UnsignedShort:
        // A 16-bit glTF image keeps its bits: the unsized GL_RGBA would have let the driver
        // pick 8, throwing away half of what the asset carries (Vulkan uploads R16G16B16A16).
        internalFormat = static_cast<unsigned int>(GL_RGBA16);
        break;
    default:
        break;
    }

    glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(internalFormat), texture.width, texture.height, NULL, format,
                 type, texture.data.data());

    glGenerateMipmap(GL_TEXTURE_2D);
    // GL_REPEAT is GL's default and was previously left implicit. It is the glTF sampler
    // default too, and the wrap KHR_texture_transform scaling depends on once a UV leaves
    // 0..1, so it is stated rather than inherited (Vulkan sets the same mode explicitly).
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, maxAnisotropy);

    glBindTexture(GL_TEXTURE_2D, 0);

    return textureId;
}

std::expected<unsigned int, std::string> OpenGLRenderer::createCubeMap(const Texture& front, const Texture& back,
                                                                       const Texture& left, const Texture& right,
                                                                       const Texture& top, const Texture& bottom)
{
    unsigned int textureId;
    glGenTextures(1, &textureId);
    createdTextures.push_back(textureId);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureId);

    const Texture* textures[] = {&right, &left, &bottom, &top, &front, &back};
    for (auto i = 0; i < 6; i++)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0,
                     getInternalFormatFromBitsPerPixel(static_cast<int>(textures[i]->bitsPerPixel)), textures[i]->width,
                     textures[i]->height, 0, getTextureFormat(textures[i]->format),
                     getTextureDataType(textures[i]->pixelDataType), textures[i]->data.data());
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, 12);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    // The faces' pixel data is CubeMapService's to drop, through the handles it holds. This used
    // to const_cast the const Texture& it was handed and empty it here — which is also what made
    // a texture shared between a cube face and a material slot upload from an empty vector, since
    // nothing wrote a gpuResourceId to say the pixels had gone.
    return textureId;
}

std::expected<unsigned int, std::string> OpenGLRenderer::createFbo(const Fbo& fbo)
{
    unsigned int fboGpuResourceId;
    glGenFramebuffers(1, &fboGpuResourceId);
    createdFbos.push_back(fboGpuResourceId);
    glBindFramebuffer(GL_FRAMEBUFFER, fboGpuResourceId);

    auto colourAttachmentIndex = 0u;
    std::vector<unsigned int> drawBuffers;

    for (const auto& attachmentKey : fbo.attachments)
    {
        memoryStorageService.bufferAttachments.mutate(
            attachmentKey,
            [&](FboAttachment& attachment)
            {
                unsigned int attachmentGpuResourceId;
                glGenTextures(1, &attachmentGpuResourceId);
                createdTextures.push_back(attachmentGpuResourceId);
                glBindTexture(GL_TEXTURE_2D, attachmentGpuResourceId);

                glTexImage2D(GL_TEXTURE_2D, 0, getTextureFormat(attachment.internalFormat), attachment.width,
                             attachment.height, 0, getTextureFormat(attachment.captureFormat), GL_FLOAT, nullptr);

                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

                // No default: every FboAttachmentType has a GL attachment point, and a new
                // enumerator must be given one here rather than discovered at run time as a
                // framebuffer that reported complete with an attachment bound to point 0.
                unsigned int attachmentType = 0;
                switch (attachment.type)
                {
                case FboAttachmentType::Color:
                    attachmentType = GL_COLOR_ATTACHMENT0 + colourAttachmentIndex;
                    colourAttachmentIndex++;
                    drawBuffers.push_back(attachmentType);
                    break;
                case FboAttachmentType::Depth:
                    attachmentType = GL_DEPTH_ATTACHMENT;
                    break;
                case FboAttachmentType::Stencil:
                    attachmentType = GL_STENCIL_ATTACHMENT;
                    break;
                }

                glFramebufferTexture2D(GL_FRAMEBUFFER, attachmentType, GL_TEXTURE_2D, attachmentGpuResourceId, 0);
                attachment.gpuResourceId = attachmentGpuResourceId;
            });
    }

    glDrawBuffers(static_cast<GLsizei>(drawBuffers.size()), drawBuffers.data());

    const auto status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        const auto* statusName = "unknown status";
        switch (status)
        {
        case GL_FRAMEBUFFER_UNDEFINED:
            statusName = "GL_FRAMEBUFFER_UNDEFINED";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            statusName = "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            statusName = "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
            statusName = "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
            statusName = "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
            break;
        case GL_FRAMEBUFFER_UNSUPPORTED:
            statusName = "GL_FRAMEBUFFER_UNSUPPORTED";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
            statusName = "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
            statusName = "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
            break;
        default:
            break;
        }

        return std::unexpected(std::string("the framebuffer is incomplete: ") + statusName + " (" +
                               std::to_string(status) + ")");
    }

    return fboGpuResourceId;
}

void OpenGLRenderer::deleteFbo(Fbo& fbo)
{
    // GL recycles a deleted name immediately, and createFbo regenerates these objects right
    // after this returns: an id left in the tracking vectors would make the destructor delete
    // whatever live object inherited the name.
    if (fbo.gpuResourceId.has_value())
    {
        releaseGpuResource(GpuResourceKind::FrameBuffer, fbo.gpuResourceId.value());
        fbo.gpuResourceId.reset();
    }

    for (const auto& attachmentKey : fbo.attachments)
    {
        memoryStorageService.bufferAttachments.mutate(attachmentKey,
                                                      [&](FboAttachment& attachment)
                                                      {
                                                          if (!attachment.gpuResourceId.has_value())
                                                          {
                                                              return;
                                                          }

                                                          releaseGpuResource(GpuResourceKind::Texture,
                                                                             attachment.gpuResourceId.value());
                                                          attachment.gpuResourceId.reset();
                                                      });
    }
}

// GL frees at the call: the driver retires a name that is still referenced by commands already
// issued, so there is nothing to defer. The tracking vector is what the destructor deletes from,
// and a name left in it after a delete would make teardown delete whatever live object inherited
// the name — which is exactly what the resize path used to do.
void OpenGLRenderer::releaseGpuResource(const GpuResourceKind kind, const unsigned int gpuResourceId)
{
    switch (kind)
    {
    case GpuResourceKind::Buffer:
        glDeleteBuffers(1, &gpuResourceId);
        std::erase(createdVbos, gpuResourceId);
        break;
    case GpuResourceKind::Texture:
    case GpuResourceKind::CubeMap:
        glDeleteTextures(1, &gpuResourceId);
        std::erase(createdTextures, gpuResourceId);
        break;
    case GpuResourceKind::VertexArray:
        glDeleteVertexArrays(1, &gpuResourceId);
        std::erase(createdVaos, gpuResourceId);
        break;
    case GpuResourceKind::ShaderProgram:
        glDeleteProgram(gpuResourceId);
        std::erase(createdPrograms, gpuResourceId);
        // Uniform locations are per program and the pool caches them — including the permanent
        // -1 for a name the program does not carry. A recycled program id would inherit them.
        std::erase_if(uniformPool, [&](const auto& entry) { return entry.first.first == gpuResourceId; });
        break;
    case GpuResourceKind::FrameBuffer:
        glDeleteFramebuffers(1, &gpuResourceId);
        std::erase(createdFbos, gpuResourceId);
        break;
    }
}

// GL keys nothing on a material handle except the latch that skips rebinding an unchanged
// material, and that latch has to drop: the next material to land in this slot is a different
// material wearing the same index.
void OpenGLRenderer::releaseMaterial(const Resource<Material>& material)
{
    if (currentlyBoundMaterial.has_value() && currentlyBoundMaterial.value() == material)
    {
        currentlyBoundMaterial.reset();
    }
}

GpuResourceCensus OpenGLRenderer::gpuResourceCensus() const
{
    return GpuResourceCensus{.buffers = static_cast<unsigned int>(createdVbos.size()),
                             .textures = static_cast<unsigned int>(createdTextures.size()),
                             .vertexArrays = static_cast<unsigned int>(createdVaos.size()),
                             .shaderPrograms = static_cast<unsigned int>(createdPrograms.size()),
                             .frameBuffers = static_cast<unsigned int>(createdFbos.size())};
}

std::expected<void, std::string> OpenGLRenderer::compileShader(const unsigned int id, const std::string& source)
{
    auto shaderSource = source.c_str();
    glShaderSource(id, 1, &shaderSource, nullptr);
    glCompileShader(id);

    auto result = GL_FALSE;
    GLint infoLogLength = 0;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &infoLogLength);

    std::string compileLog;
    if (infoLogLength > 1)
    {
        std::vector<char> infoLog(static_cast<size_t>(infoLogLength) + 1);
        glGetShaderInfoLog(id, infoLogLength, nullptr, infoLog.data());
        compileLog = infoLog.data();
    }

    if (result != GL_TRUE)
    {
        return std::unexpected(compileLog.empty() ? std::string("the stage did not compile") : compileLog);
    }

    // A log on a stage that compiled is a driver remark, not a failure, and there is no caller
    // decision attached to it.
    if (!compileLog.empty())
    {
        logger.warn("Shader compile log: {}", compileLog);
    }

    return {};
}

int OpenGLRenderer::getUniformLocation(const unsigned int programId, const char* uniformName)
{
    const auto key = std::pair<unsigned int, std::string_view>(programId, uniformName);
    const auto result = uniformPool.find(key);

    if (result != uniformPool.end())
    {
        return result->second;
    }

    const auto propertyLocation = glGetUniformLocation(programId, uniformName);
    uniformPool.emplace(std::pair<unsigned int, std::string>(programId, uniformName), propertyLocation);

    return propertyLocation;
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const char* uniformName, const int value)
{
    glProgramUniform1i(programId, getUniformLocation(programId, uniformName), value);
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const char* uniformName, const float data)
{
    glProgramUniform1f(programId, getUniformLocation(programId, uniformName), data);
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const char* uniformName, const double data)
{
    glProgramUniform1d(programId, getUniformLocation(programId, uniformName), data);
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const char* uniformName, const glm::vec2& data)
{
    glProgramUniform2f(programId, getUniformLocation(programId, uniformName), data.x, data.y);
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const char* uniformName, const glm::vec3& data)
{
    glProgramUniform3f(programId, getUniformLocation(programId, uniformName), data.x, data.y, data.z);
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const char* uniformName, const glm::vec4& data)
{
    glProgramUniform4f(programId, getUniformLocation(programId, uniformName), data.x, data.y, data.z, data.w);
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const char* uniformName, const glm::mat3& data)
{
    glProgramUniformMatrix3fv(programId, getUniformLocation(programId, uniformName), 1, GL_FALSE, &data[0][0]);
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const char* uniformName, const glm::mat4& data)
{
    auto location = getUniformLocation(programId, uniformName);
    glProgramUniformMatrix4fv(programId, location, 1, GL_FALSE, &data[0][0]);
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const char* uniformName,
                                       const std::vector<glm::vec2>& data)
{
    glProgramUniform2fv(programId, getUniformLocation(programId, uniformName), static_cast<GLsizei>(data.size()),
                        reinterpret_cast<const float*>(data.data()));
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const char* uniformName,
                                       const std::vector<glm::vec3>& data)
{
    glProgramUniform3fv(programId, getUniformLocation(programId, uniformName), static_cast<GLsizei>(data.size()),
                        reinterpret_cast<const float*>(data.data()));
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const char* uniformName,
                                       const std::vector<glm::vec4>& data)
{
    glProgramUniform4fv(programId, getUniformLocation(programId, uniformName), static_cast<GLsizei>(data.size()),
                        reinterpret_cast<const float*>(data.data()));
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const char* uniformName,
                                       const std::vector<glm::mat3>& data)
{
    glProgramUniformMatrix3fv(programId, getUniformLocation(programId, uniformName), static_cast<GLsizei>(data.size()),
                              GL_FALSE, reinterpret_cast<const float*>(data.data()));
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const char* uniformName,
                                       const std::vector<glm::mat4>& data)
{
    setProgramUniform(programId, uniformName, data, data.size());
}

// Uploading fewer elements than the vector holds is how the joint palette is clamped to the
// shader's array bound without copying it.
void OpenGLRenderer::setProgramUniform(const unsigned int programId, const char* uniformName,
                                       const std::vector<glm::mat4>& data, const size_t count)
{
    glProgramUniformMatrix4fv(programId, getUniformLocation(programId, uniformName), static_cast<GLsizei>(count),
                              GL_FALSE, reinterpret_cast<const float*>(data.data()));
}

void OpenGLRenderer::setViewport(int width, int height)
{
    viewportWidth = width;
    viewportHeight = height;
    glViewport(0, 0, width, height);
}

std::expected<void, std::string> OpenGLRenderer::captureFrame(const std::string& path)
{
    // viewportWidth/Height track the window size: set from the window state at startup
    // and updated by the resize callback before any capture can run.
    const auto width = viewportWidth;
    const auto height = viewportHeight;
    const auto rowBytes = static_cast<size_t>(width) * 4;

    auto pixels = std::vector<unsigned char>(rowBytes * static_cast<size_t>(height));

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    auto row = std::vector<unsigned char>(rowBytes);
    for (auto y = 0; y < height / 2; y++)
    {
        auto* top = pixels.data() + static_cast<size_t>(y) * rowBytes;
        auto* bottom = pixels.data() + static_cast<size_t>(height - 1 - y) * rowBytes;
        std::memcpy(row.data(), top, rowBytes);
        std::memcpy(top, bottom, rowBytes);
        std::memcpy(bottom, row.data(), rowBytes);
    }

    if (stbi_write_png(path.c_str(), width, height, 4, pixels.data(), static_cast<int>(rowBytes)) == 0)
    {
        return std::unexpected("stb_image_write could not write the frame dump to " + path);
    }

    logger.info("Frame dump written to {}", path);
    return {};
}

unsigned int OpenGLRenderer::getTextureDataType(PixelDataType type) const
{
    switch (type)
    {
    case PixelDataType::UnsignedByte:
        return GL_UNSIGNED_BYTE;
    case PixelDataType::UnsignedShort:
        return GL_UNSIGNED_SHORT;
    case PixelDataType::Float:
        return GL_FLOAT;
    default:
        return GL_UNSIGNED_BYTE;
    }
}

unsigned int OpenGLRenderer::getTextureFormat(TextureFormat format) const
{
    switch (format)
    {
    case TextureFormat::R:
        return GL_RED;
    case TextureFormat::RG:
        return GL_RG;
    case TextureFormat::RGB:
        return GL_RGB;
    case TextureFormat::RGBA:
        return GL_RGBA;
    case TextureFormat::RGBA16F:
        return GL_RGBA16F;
    case TextureFormat::RGBA32F:
        return GL_RGBA32F;
    case TextureFormat::DepthComponent:
        return GL_DEPTH_COMPONENT;
    default:
        return GL_RGB;
    }
}

unsigned int OpenGLRenderer::getInternalFormatFromBitsPerPixel(int bitsPerPixel) const
{
    switch (bitsPerPixel)
    {
    case 128:
        return GL_RGBA32F;
    case 96:
        return GL_RGB32F;
    case 32:
        return GL_RGBA;
    case 24:
        return GL_RGB;
    case 16:
        return GL_RG;
    case 8:
        return GL_RED;
    default:
        return GL_RGB;
    }
}

void OpenGLRenderer::createQuad()
{
    vertices = std::vector<float>({-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
                                   0.0,  0.0,  1.0, 0.0,  1.0, 1.0, 1.0, 1.0, 0.0,  1.0, 0.0,  0.0});

    glGenVertexArrays(1, &quadVao);
    glBindVertexArray(quadVao);
    createdVaos.push_back(quadVao);

    GLuint vertexBuffer;
    glGenBuffers(1, &vertexBuffer);
    createdVbos.push_back(vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<const void*>(12 * sizeof(float)));

    glBindVertexArray(0);
}

bool OpenGLRenderer::drawFullScreenQuad(const Resource<Shader>& shaderKey,
                                        const std::vector<Resource<FboAttachment>>& attachmentKeys) const
{
    const auto* shader = memoryStorageService.shaders.find(shaderKey);
    if (shader == nullptr)
    {
        return false;
    }

    glViewport(0, 0, viewportWidth, viewportHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shader->gpuResourceId);

    for (size_t i = 0; i < attachmentKeys.size(); i++)
    {
        const auto* attachment = memoryStorageService.bufferAttachments.find(attachmentKeys[i]);

        if (attachment != nullptr && attachment->gpuResourceId.has_value())
        {
            glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(i));
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(attachment->gpuResourceId.value()));
        }
    }

    glBindVertexArray(quadVao);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size() / 4));

    return true;
}

} // namespace raceengine
