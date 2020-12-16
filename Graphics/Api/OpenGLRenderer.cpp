#include "OpenGLRenderer.h"
#include <gl/gl3w.h>
#include <vector>

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

    glClearColor(0.f / 255.f, 0.f / 255.f, 0.f / 255.f, 0);

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
    if (currentlyBoundMaterial != mesh.material) {
        bindMaterial(mesh.material);
        currentlyBoundMaterial = mesh.material;
    }

    drawPrimitives(mesh.mesh->meshPrimitives);
}

void OpenGLRenderer::drawPrimitives(const std::vector<MeshPrimitive>& primitives) const
{
    for (const auto& primitive : primitives)
    {
        glDrawElements(primitive.mode,
                       static_cast<GLsizei>(primitive.elementCount),
                       primitive.indicesType == VertexIndicesType::UnsignedInt ? GL_UNSIGNED_INT :
                       primitive.indicesType == VertexIndicesType::UnsignedShort ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE,
                       primitive.indicesOffset);
    }
}

void OpenGLRenderer::draw(const Scene* scene, float delta)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    auto camera = sceneService.getCamera(scene, 0);
    cameraService.updateModelViewProjectionMatrix(camera);

    for (auto& sky : scene->skybox) {
        glBindVertexArray(static_cast<GLuint>(sky->model->gpuResourceId));

        for (auto& mesh : sky->model->meshes)
        {
            glUseProgram(sky->shader->gpuResourceId);

            setProgramUniform(sky->shader->gpuResourceId, "localToScreen4x4Matrix",
                              camera->modelViewProjectionMatrix *  sceneManagerService.modelMatrix(sky->node));

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_CUBE_MAP, static_cast<GLuint>(sky->cubeMap->gpuResourceId));

            drawPrimitives(mesh.meshPrimitives);
        }
    }

    for (auto& entity : scene->models)
    {
        auto renderableModel = reinterpret_cast<RenderableModel*>(entity.get());
        auto model = renderableModel->model;
        glBindVertexArray(static_cast<GLuint>(model->gpuResourceId));

        for (auto& mesh : renderableModel->meshes)
        {
            if (mesh.material == nullptr)
            {
                continue;
            }

            const auto entityModelMatrix = sceneManagerService.modelMatrix(entity->node);

            setProgramUniform(mesh.material->shader->gpuResourceId, "cameraPosition", camera->position);
            setProgramUniform(mesh.material->shader->gpuResourceId, "localToScreen4x4Matrix",
                              camera->modelViewProjectionMatrix * entityModelMatrix);
            setProgramUniform(mesh.material->shader->gpuResourceId, "localToView4x4Matrix",
                              camera->modelViewMatrix * entityModelMatrix);
            setProgramUniform(mesh.material->shader->gpuResourceId, "localToWorld4x4Matrix", entityModelMatrix);
            setProgramUniform(mesh.material->shader->gpuResourceId, "modelView3x3Matrix",
                              glm::mat3(camera->modelViewMatrix));
            setProgramUniform(mesh.material->shader->gpuResourceId, "normalMatrix",
                              glm::transpose(glm::inverse(glm::mat3(camera->modelViewMatrix * entityModelMatrix))));
            const auto joints = renderableEntityService.joints(mesh, delta);
            setProgramUniform(mesh.material->shader->gpuResourceId, "jointTransformationMatrixes", joints);
            setProgramUniform(mesh.material->shader->gpuResourceId, "animated", !joints.empty());

            setProgramUniform(mesh.material->shader->gpuResourceId, "textureRepeat", mesh.material->repeat);

            setProgramUniform(mesh.material->shader->gpuResourceId, "lights.position",
                              glm::vec3(0.0f, 350.0f, 350.0f));
            setProgramUniform(mesh.material->shader->gpuResourceId, "lights.diffuse",
                              glm::vec3(1.2859 * 2.5f, 1.2973 * 2.5f, 1.3 * 2.5f));
            setProgramUniform(mesh.material->shader->gpuResourceId, "lights.specular",
                              glm::vec3(1.2859, 1.2973, 1.3));
            setProgramUniform(mesh.material->shader->gpuResourceId, "lights.ambient",
                              glm::vec3(0.29859, 0.29973, 0.3));
            setProgramUniform(mesh.material->shader->gpuResourceId, "lights.attenuation", 1.0f);

            drawMesh(mesh);
        }
    }
}

void OpenGLRenderer::bindMaterial(const Material* material)
{
    glUseProgram(material->shader->gpuResourceId);

    if (material->albedo.has_value() && material->albedo.value()->gpuResourceId.has_value())
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(material->albedo.value()->gpuResourceId.value()));
    }

    if (material->normal.has_value() && material->normal.value()->gpuResourceId.has_value())
    {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(material->normal.value()->gpuResourceId.value()));
    }

    if (material->metallicRoughness.has_value() && material->metallicRoughness.value()->gpuResourceId.has_value())
    {
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(material->metallicRoughness.value()->gpuResourceId.value()));
    }

    if (material->emissive.has_value() && material->emissive.value()->gpuResourceId.has_value())
    {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(material->emissive.value()->gpuResourceId.value()));

    }

    if (material->occlusion.has_value() && material->occlusion.value()->gpuResourceId.has_value())
    {
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(material->occlusion.value()->gpuResourceId.value()));
    }

    for (auto i = 0; i < material->textures.size(); i++)
    {
        if (material->textures[i]->gpuResourceId.has_value())
        {
            glActiveTexture(GL_TEXTURE4 + i + 1);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(material->textures[i]->gpuResourceId.value()));
        }
    }
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

    for (auto& mesh : model->meshes)
    {
        for (auto material : mesh.materials)
        {
            if (material->albedo.has_value() && !material->albedo.value()->gpuResourceId.has_value())
            {
                createTexture(material->albedo.value());
            }

            if (material->occlusion.has_value() && !material->occlusion.value()->gpuResourceId.has_value())
            {
                createTexture(material->occlusion.value());
            }

            if (material->emissive.has_value() && !material->emissive.value()->gpuResourceId.has_value())
            {
                createTexture(material->emissive.value());
            }

            if (material->metallicRoughness.has_value() &&
                !material->metallicRoughness.value()->gpuResourceId.has_value())
            {
                createTexture(material->metallicRoughness.value());
            }

            if (material->normal.has_value() && !material->normal.value()->gpuResourceId.has_value())
            {
                createTexture(material->normal.value());
            }
        }
    }
}

std::optional<unsigned int> OpenGLRenderer::createShaderObject(const ShaderDescriptor& object)
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
        if (compileShader(vertexShaderId, object.vertexShaderSource))
        {
            glAttachShader(programId, vertexShaderId);
        }
    }

    if (!object.fragmentShaderSource.empty())
    {
        if (compileShader(fragmentShaderId, object.fragmentShaderSource))
        {
            glAttachShader(programId, fragmentShaderId);
        }
    }

    if (!object.tessellationControlShaderSource.empty() && !object.tessellationEvaluationShaderSource.empty())
    {
        if (compileShader(tessellationControlShaderId, object.tessellationControlShaderSource) &&
            compileShader(tessellationEvaluationShaderId, object.tessellationEvaluationShaderSource))
        {
            glAttachShader(programId, tessellationControlShaderId);
            glAttachShader(programId, tessellationEvaluationShaderId);
        }
    }

    if (!object.computeShaderSource.empty())
    {
        if (compileShader(computeShaderId, object.computeShaderSource))
        {
            glAttachShader(programId, computeShaderId);
        }
    }

    if (!object.geometryShaderSource.empty())
    {
        if (compileShader(geometryShaderId, object.geometryShaderSource))
        {
            glAttachShader(programId, geometryShaderId);
        }
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
        logger.error("shader link error: {}", std::string(errorMessage.begin(), errorMessage.end()));

        return {};
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

    auto format = getTextureFormat(texture->format);
    auto type = getTextureDataType(texture->pixelDataType);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture->width, texture->height, NULL, format, type, texture->data.data());

    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
    glTexParameterf(GL_TEXTURE_2D, GL_MAX_TEXTURE_MAX_ANISOTROPY, fLargest);

    glBindTexture(GL_TEXTURE_2D, 0);

    texture->gpuResourceId = textureId;

    return textureId;
}

unsigned int OpenGLRenderer::createCubeMap(
    Texture* front, Texture* back, Texture* left, Texture* right, Texture* top, Texture* bottom) const
{
    unsigned int textureId;
    glGenTextures(1, &textureId);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureId);

    Texture* textures[] = { right, left, bottom, top, front, back };
    for (auto i = 0; i < 6; i++)
    {
        glTexImage2D(
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
            0,
            getInternalFormatFromBitsPerPixel(
            textures[i]->bitsPerPixel),
            textures[i]->width,
            textures[i]->height,
            0,
            getTextureFormat(textures[i]->format),
            getTextureDataType(textures[i]->pixelDataType),
            textures[i]->data.data());
    }

    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, 6);

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

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
        case TextureFormat::BGR:
            return GL_BGR;
        case TextureFormat::BGRA:
            return GL_BGRA;
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