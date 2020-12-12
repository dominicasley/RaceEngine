#include "OpenGLRenderer.h"
#include <gl/gl3w.h>
#include <vector>
#include "../Models/Scene/RenderableEntityDesc.h"
#include "Resources/Shader.h"
#include "../Models/Scene/Scene.h"
#include "../Models/Scene/RenderableEntity.h"
#include "../Models/Scene/Light.h"
#include "../Models/Scene/Camera.h"
#include "../Models/Scene/Model.h"

OpenGLRenderer::OpenGLRenderer(
    spdlog::logger& logger,
    RenderableEntityService& renderableEntityService,
    SceneService& sceneService,
    SceneManagerService& sceneManagerService,
    CameraService& cameraService) :
    logger(logger),
    renderableEntityService(renderableEntityService),
    sceneService(sceneService),
    sceneManagerService(sceneManagerService),
    cameraService(cameraService)
{
}

bool OpenGLRenderer::init()
{
    if (gl3wInit())
    {
        logger.error("Failed to initialize OpenGL");
        return false;
    }

    glClearColor(100.f / 255.f, 149.f / 255.f, 237.f / 255.f, 0);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_3D);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_TEXTURE_CUBE_MAP);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glEnable(GL_CULL_FACE);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glCullFace(GL_BACK);

    return true;
}

void OpenGLRenderer::drawMesh(const RenderableMesh& mesh)
{
    for (const auto& primitive : mesh.mesh->meshPrimitives)
    {
        if (primitive.material.albedo.has_value() &&
            primitive.material.albedo.value()->gpuResourceId.has_value())
        {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(
                GL_TEXTURE_2D,
                static_cast<GLuint>(primitive.material.albedo.value()->gpuResourceId.value()));
        }

        if (primitive.material.normal.has_value() &&
            primitive.material.normal.value()->gpuResourceId.has_value())
        {
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(
                GL_TEXTURE_2D,
                static_cast<GLuint>(primitive.material.normal.value()->gpuResourceId.value()));
        }

        if (primitive.material.metallicRoughness.has_value() &&
            primitive.material.metallicRoughness.value()->gpuResourceId.has_value())
        {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(
                GL_TEXTURE_2D,
                static_cast<GLuint>(primitive.material.metallicRoughness.value()->gpuResourceId.value()));
        }

        if (primitive.material.emissive.has_value() &&
            primitive.material.emissive.value()->gpuResourceId.has_value())
        {
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(
                GL_TEXTURE_2D,
                static_cast<GLuint>(primitive.material.emissive.value()->gpuResourceId.value()));
        }

        if (primitive.material.occlusion.has_value() &&
            primitive.material.occlusion.value()->gpuResourceId.has_value())
        {
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(
                GL_TEXTURE_2D,
                static_cast<GLuint>(primitive.material.occlusion.value()->gpuResourceId.value()));
        }

        glDrawElements(primitive.mode,
                       static_cast<GLsizei>(primitive.elementCount),
                       primitive.indicesType == VertexIndicesType::UnsignedInt ? GL_UNSIGNED_INT :
                       primitive.indicesType == VertexIndicesType::UnsignedShort ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE,
                       primitive.indicesOffset);
    }
}

void OpenGLRenderer::draw(const Scene& scene)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    auto camera = sceneService.getCamera(scene, 0);
    cameraService.updateModelViewProjectionMatrix(*camera);

    glUseProgram(1);
    setProgramUniform(1, "cameraPosition", camera->position);

    for (auto& entity : scene.entities)
    {
        const auto entityModelMatrix = sceneManagerService.modelMatrix(entity->node);

        setProgramUniform(1, "localToScreen4x4Matrix", camera->modelViewProjectionMatrix * entityModelMatrix);
        setProgramUniform(1, "localToWorld4x4Matrix", entityModelMatrix);
        setProgramUniform(1, "localToWorld3x3Matrix", glm::mat3(entityModelMatrix));
        setProgramUniform(1, "normalMatrix",
                          glm::transpose(glm::inverse(glm::mat3(camera->modelViewMatrix * entityModelMatrix))));

        auto model = entity->model;
        glBindVertexArray(static_cast<GLuint>(model->gpuResourceId));
        for (auto& mesh : entity->meshes)
        {
            const auto joints = renderableEntityService.joints(mesh);
            setProgramUniform(1, "jointTransformationMatrixes", joints);
            setProgramUniform(1, "animated", !joints.empty());
            drawMesh(mesh);
        }
    }
}

void OpenGLRenderer::bindMaterial(const Material* material)
{
    // glUseProgram(material->getShader());

    // auto offset = 0;
    // for (const auto& texture : material->getTextures())
    // {
    //     if (texture)
    //     {
    //         glActiveTexture(GL_TEXTURE0 + offset);
    //         glBindTexture(GL_TEXTURE_2D, texture);
    //     }

    //     offset++;
    // }
}

std::vector<unsigned int>& OpenGLRenderer::bindMesh(std::vector<unsigned int>& vbos, const Mesh& mesh)
{
    for (const auto& buffer : mesh.meshBuffers)
    {
        if (buffer.target == 0)
        {
            logger.warn("bufferView.target is zero. skipping buffer view.");
            continue;
        }

        GLuint vbo;
        glGenBuffers(1, &vbo);
        vbos.push_back(vbo);

        glBindBuffer(buffer.target, vbo);
        glBufferData(buffer.target, buffer.length, buffer.data.data() + buffer.offset, GL_STATIC_DRAW);
    }

    for (const auto& primitive : mesh.meshPrimitives)
    {
        glBindBuffer(GL_ARRAY_BUFFER, vbos[primitive.bufferIndex]);

        if (primitive.attributeType.has_value())
        {
            glEnableVertexAttribArray(static_cast<GLuint>(primitive.attributeType.value()));
            glVertexAttribPointer(static_cast<GLuint>(primitive.attributeType.value()),
                                  primitive.size,
                                  primitive.componentType,
                                  primitive.normalized ? GL_TRUE : GL_FALSE,
                                  primitive.stride,
                                  reinterpret_cast<void*>(primitive.offset));
        }
    }

    return vbos;
}

void OpenGLRenderer::upload(Model* model)
{
    GLuint vao;
    std::vector<unsigned int> vbos;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    for (const auto& mesh : model->meshes)
    {
        bindMesh(vbos, mesh);
    }

    glBindVertexArray(0);

    for (unsigned int& vbo : vbos)
    {
        glDeleteBuffers(1, &vbo);
    }

    model->gpuResourceId = static_cast<int>(vao);

    for (auto& mesh : model->meshes) {
        for (auto& primitive : mesh.meshPrimitives) {
            if (primitive.material.albedo.has_value() && !primitive.material.albedo.value()->gpuResourceId.has_value()) {
                createTexture(primitive.material.albedo.value());
            }
        }
    }
}

unsigned int OpenGLRenderer::createShaderObject(const Shader& object)
{
    const auto programId = glCreateProgram();
    const auto vertexShaderId = glCreateShader(GL_VERTEX_SHADER);
    const auto fragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);
    const auto tessellationControlShaderId = glCreateShader(GL_TESS_CONTROL_SHADER);
    const auto tessellationEvaluationShaderId = glCreateShader(GL_TESS_EVALUATION_SHADER);
    const auto computeShaderId = glCreateShader(GL_COMPUTE_SHADER);
    const auto geometryShaderId = glCreateShader(GL_GEOMETRY_SHADER);

    if (!object.vertexShaderSource.empty())
    {
        compileShader(vertexShaderId, object.vertexShaderSource);
        glAttachShader(programId, vertexShaderId);
    }

    if (!object.fragmentShaderSource.empty())
    {
        compileShader(fragmentShaderId, object.fragmentShaderSource);
        glAttachShader(programId, fragmentShaderId);
    }

    if (!object.tessellationControlShaderSource.empty() && !object.tessellationEvaluationShaderSource.empty())
    {
        compileShader(tessellationControlShaderId, object.tessellationControlShaderSource);
        compileShader(tessellationEvaluationShaderId, object.tessellationEvaluationShaderSource);
        glAttachShader(programId, tessellationControlShaderId);
        glAttachShader(programId, tessellationEvaluationShaderId);
    }

    if (!object.computeShaderSource.empty())
    {
        compileShader(computeShaderId, object.computeShaderSource);
        glAttachShader(programId, computeShaderId);
    }

    if (!object.geometryShaderSource.empty())
    {
        compileShader(geometryShaderId, object.geometryShaderSource);
        glAttachShader(programId, geometryShaderId);
    }

    glLinkProgram(programId);

    auto result = GL_FALSE;
    GLint infoLogLength;
    glGetProgramiv(programId, GL_LINK_STATUS, &result);
    glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &infoLogLength);

    if (infoLogLength > 1)
    {
        std::vector<char> errorMessage(infoLogLength + 1);
        glGetProgramInfoLog(programId, infoLogLength, nullptr, &errorMessage[0]);
        fprintf(stderr, "%s", &errorMessage[0]);
    }

    glDeleteShader(vertexShaderId);
    glDeleteShader(fragmentShaderId);
    glDeleteShader(tessellationControlShaderId);
    glDeleteShader(tessellationEvaluationShaderId);
    glDeleteShader(computeShaderId);
    glDeleteShader(geometryShaderId);

    return programId;
}

unsigned int OpenGLRenderer::createTexture(Texture* texture) const
{
    unsigned int textureId;

    GLfloat fLargest;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &fLargest);

    glGenTextures(1, &textureId);
    glBindTexture(GL_TEXTURE_2D, textureId);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    GLenum format =
        texture->format == TextureFormat::R ? GL_RED :
        texture->format == TextureFormat::RG ? GL_RG :
        TextureFormat::RGB == 3 ? GL_RGB : GL_RGBA;

    GLenum type = texture->pixelDataType == PixelDataType::UnsignedShort ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE;

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture->width, texture->height, NULL, format, type, texture->data.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    texture->gpuResourceId = textureId;
    return textureId;
}

bool OpenGLRenderer::compileShader(const unsigned int id, const std::string& source)
{
    auto vertexShader = source.c_str();
    glShaderSource(id, 1, &vertexShader, nullptr);
    glCompileShader(id);

    auto result = GL_FALSE;
    GLint infoLogLength;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &infoLogLength);

    if (infoLogLength > 1)
    {
        std::vector<char> errorMessage(infoLogLength + 1);
        glGetShaderInfoLog(id, infoLogLength, nullptr, &errorMessage[0]);
        logger.error("Shader compile error: {}", errorMessage.data());

        return false;
    }

    return true;
}

unsigned int OpenGLRenderer::getUniformLocation(unsigned int programId, const std::string& uniformName)
{
    unsigned int propertyLocation;
    const auto result = uniformPool.find(UniformKey(programId, uniformName));

    if (result != uniformPool.end())
    {
        propertyLocation = result->second;
    }
    else
    {
        propertyLocation = glGetUniformLocation(programId, uniformName.c_str());
        uniformPool[UniformKey(programId, uniformName)] = propertyLocation;
    }

    return propertyLocation;
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const std::string& uniformName, const int value)
{
    glProgramUniform1i(programId, getUniformLocation(programId, uniformName), value);
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const std::string& uniformName, const float data)
{
    glProgramUniform1f(programId, getUniformLocation(programId, uniformName), data);
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const std::string& uniformName, const double data)
{
    glProgramUniform1d(programId, getUniformLocation(programId, uniformName), data);
}

void
OpenGLRenderer::setProgramUniform(const unsigned int programId, const std::string& uniformName, const glm::vec2& data)
{
    glProgramUniform2f(programId, getUniformLocation(programId, uniformName), data.x, data.y);
}

void
OpenGLRenderer::setProgramUniform(const unsigned int programId, const std::string& uniformName, const glm::vec3& data)
{
    glProgramUniform3f(programId, getUniformLocation(programId, uniformName), data.x, data.y, data.z);
}

void
OpenGLRenderer::setProgramUniform(const unsigned int programId, const std::string& uniformName, const glm::vec4& data)
{
    glProgramUniform4f(programId, getUniformLocation(programId, uniformName), data.x, data.y, data.z, data.w);
}

void
OpenGLRenderer::setProgramUniform(const unsigned int programId, const std::string& uniformName, const glm::mat3& data)
{
    glProgramUniformMatrix3fv(programId, getUniformLocation(programId, uniformName), 1, GL_FALSE, &data[0][0]);
}

void
OpenGLRenderer::setProgramUniform(const unsigned int programId, const std::string& uniformName, const glm::mat4& data)
{
    auto location = getUniformLocation(programId, uniformName);
    glProgramUniformMatrix4fv(programId, location, 1, GL_FALSE, &data[0][0]);
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const std::string& uniformName,
                                       const std::vector<glm::vec2>& data)
{
    glProgramUniform2fv(programId, getUniformLocation(programId, uniformName), static_cast<GLsizei>(data.size()),
                        (float*) data.data());
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const std::string& uniformName,
                                       const std::vector<glm::vec3>& data)
{
    glProgramUniform3fv(programId, getUniformLocation(programId, uniformName), static_cast<GLsizei>(data.size()),
                        (float*) data.data());
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const std::string& uniformName,
                                       const std::vector<glm::vec4>& data)
{
    glProgramUniform4fv(programId, getUniformLocation(programId, uniformName), static_cast<GLsizei>(data.size()),
                        (float*) data.data());
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const std::string& uniformName,
                                       const std::vector<glm::mat3>& data)
{
    glProgramUniformMatrix3fv(programId, getUniformLocation(programId, uniformName), static_cast<GLsizei>(data.size()),
                              GL_FALSE, (float*) data.data());
}

void OpenGLRenderer::setProgramUniform(const unsigned int programId, const std::string& uniformName,
                                       const std::vector<glm::mat4>& data)
{
    glProgramUniformMatrix4fv(programId, getUniformLocation(programId, uniformName), static_cast<GLsizei>(data.size()),
                              GL_FALSE, (float*) data.data());
}

void OpenGLRenderer::setViewport(int width, int height)
{
    glViewport(0, 0, width, height);
}
