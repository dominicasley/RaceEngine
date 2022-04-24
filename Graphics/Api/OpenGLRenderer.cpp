#include "OpenGLRenderer.h"
#include <gl/gl3w.h>
#include <vector>
#include <ranges>

OpenGLRenderer::OpenGLRenderer(
    spdlog::logger& logger,
    RenderableEntityService& renderableEntityService,
    SceneManagerService& sceneManagerService,
    MemoryStorageService& memoryStorageService) :
    logger(logger),
    renderableEntityService(renderableEntityService),
    sceneManagerService(sceneManagerService),
    memoryStorageService(memoryStorageService)
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

    createQuad();

    return true;
}

void OpenGLRenderer::drawMesh(const RenderableMesh& renderableMesh)
{
    if (!currentlyBoundMaterial.has_value() || currentlyBoundMaterial.value().id != renderableMesh.material.value().id)
    {
        bindMaterial(renderableMesh.material.value());
        currentlyBoundMaterial = renderableMesh.material.value();
    }

    drawPrimitives(renderableMesh.mesh->meshPrimitives);
}

void OpenGLRenderer::drawPrimitives(const std::vector<MeshPrimitive>& primitives) const
{
    for (const auto& primitive: primitives)
    {
        glDrawElements(primitive.mode,
                       static_cast<GLsizei>(primitive.elementCount),
                       primitive.indicesType == VertexIndicesType::UnsignedInt ? GL_UNSIGNED_INT :
                       primitive.indicesType == VertexIndicesType::UnsignedShort ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE,
                       primitive.indicesOffset);
    }
}

void OpenGLRenderer::draw(const Scene* scene, const Camera* camera, float delta)
{
    if (camera->output.has_value())
    {
        glBindFramebuffer(GL_FRAMEBUFFER, camera->output.value()->gpuResourceId.value());
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for (auto& sky: scene->skybox)
    {
        auto model = sky->model;

        if (!model->gpuResourceId.has_value()) {
            upload(model);
        }

        glBindVertexArray(static_cast<GLuint>(model->gpuResourceId.value()));

        for (auto& mesh: model->meshes)
        {
            auto shader = sky->shader;
            auto cubeMap = sky->cubeMap;

            glUseProgram(shader->gpuResourceId);

            setProgramUniform(shader->gpuResourceId, "localToScreen4x4Matrix",
                              camera->modelViewProjectionMatrix * sceneManagerService.modelMatrix(sky->node));

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_CUBE_MAP, static_cast<GLuint>(cubeMap->gpuResourceId));

            drawPrimitives(mesh->meshPrimitives);
        }
    }

    for (auto& entity: scene->models)
    {
        auto renderableModel = entity.get();
        auto model = renderableModel->model;

        if (!model->gpuResourceId.has_value()) {
            upload(model);
        }

        glBindVertexArray(static_cast<GLuint>(model->gpuResourceId.value()));

        for (auto& mesh: renderableModel->meshes)
        {
            if (!mesh.material.has_value())
            {
                continue;
            }

            auto material = mesh.material;

            if (material.value()->shader.has_value())
            {
                auto shader = material.value()->shader.value();
                auto entityModelMatrix = sceneManagerService.modelMatrix(entity->node);
                const auto joints = renderableEntityService.joints(mesh, delta);

                setProgramUniform(shader->gpuResourceId, "cameraPosition", camera->position);
                setProgramUniform(shader->gpuResourceId, "localToScreen4x4Matrix",
                                  camera->modelViewProjectionMatrix * entityModelMatrix);
                setProgramUniform(shader->gpuResourceId, "localToView4x4Matrix",
                                  camera->modelViewMatrix * entityModelMatrix);
                setProgramUniform(shader->gpuResourceId, "localToWorld4x4Matrix", entityModelMatrix);
                setProgramUniform(shader->gpuResourceId, "modelView3x3Matrix",
                                  glm::mat3(camera->modelViewMatrix));
                setProgramUniform(shader->gpuResourceId, "normalMatrix",
                                  glm::transpose(glm::inverse(glm::mat3(camera->modelViewMatrix * entityModelMatrix))));
                setProgramUniform(shader->gpuResourceId, "jointTransformationMatrixes", joints);
                setProgramUniform(shader->gpuResourceId, "animated", !joints.empty());

                setProgramUniform(shader->gpuResourceId, "textureRepeat", material.value()->repeat);

                setProgramUniform(shader->gpuResourceId, "lights.position",
                                  glm::vec3(0.0f, 350.0f, 350.0f));
                setProgramUniform(shader->gpuResourceId, "lights.diffuse",
                                  glm::vec3(1.2859 * 2.5f, 1.2973 * 2.5f, 1.3 * 2.5f));
                setProgramUniform(shader->gpuResourceId, "lights.specular",
                                  glm::vec3(1.2859, 1.2973, 1.3));
                setProgramUniform(shader->gpuResourceId, "lights.ambient",
                                  glm::vec3(0.29859, 0.29973, 0.3));
                setProgramUniform(shader->gpuResourceId, "lights.attenuation", 1.0f);
            }

            drawMesh(mesh);
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    for (const auto& postProcess: camera->postProcesses)
    {
        if (postProcess->output.has_value())
        {
            auto frameBuffer = postProcess->output.value();

            if (frameBuffer->gpuResourceId.has_value())
            {
                glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer->gpuResourceId.value());
            }
        }

        drawFullScreenQuad(postProcess->shader, postProcess->inputs);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

void OpenGLRenderer::bindMaterial(const Resource<Material>& material)
{
    auto shader = material->shader.value();

    glUseProgram(shader->gpuResourceId);

    if (material->albedo.has_value())
    {
        auto albedo = material->albedo.value();

        if (albedo->gpuResourceId.has_value())
        {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(albedo->gpuResourceId.value()));
        }
    }

    if (material->normal.has_value())
    {
        auto normal = material->normal.value();

        if (normal->gpuResourceId.has_value())
        {
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(normal->gpuResourceId.value()));
        }
    }

    if (material->metallicRoughness.has_value())
    {
        auto metallicRoughness = material->metallicRoughness.value();

        if (metallicRoughness->gpuResourceId.has_value())
        {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(metallicRoughness->gpuResourceId.value()));
        }
    }

    if (material->emissive.has_value())
    {
        auto emissive = memoryStorageService.textures.get(material->emissive.value());

        if (emissive.gpuResourceId.has_value())
        {
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(emissive.gpuResourceId.value()));
        }

    }

    if (material->occlusion.has_value())
    {
        auto occlusion = material->occlusion.value();

        if (occlusion->gpuResourceId.has_value())
        {
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(occlusion->gpuResourceId.value()));
        }
    }

    if (material->environment.has_value())
    {
        auto environment = material->environment.value();

        if (environment->gpuResourceId.has_value())
        {
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_CUBE_MAP, static_cast<GLuint>(environment->gpuResourceId.value()));
        }
    }

    for (auto i = 0; i < material->textures.size(); i++)
    {
        auto texture = material->textures[i];

        if (texture->gpuResourceId.has_value())
        {
            glActiveTexture(GL_TEXTURE5 + i + 1);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(texture->gpuResourceId.value()));
        }
    }
}

std::vector<unsigned int>& OpenGLRenderer::bindMesh(std::vector<unsigned int>& vbos, const Mesh& mesh)
{
    for (const auto& buffer: mesh.meshBuffers)
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

    for (const auto& primitive: mesh.meshPrimitives)
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

void OpenGLRenderer::upload(const Resource<Model>& modelKey)
{
    auto model = memoryStorageService.models.get(modelKey);

    if (model.gpuResourceId.has_value())
        return;

    GLuint vao;
    std::vector<unsigned int> vbos;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    for (const auto& mesh: model.meshes)
    {
        bindMesh(vbos, memoryStorageService.meshes.get(mesh));
    }

    glBindVertexArray(0);

    for (unsigned int& vbo: vbos)
    {
        glDeleteBuffers(1, &vbo);
    }

    model.gpuResourceId = static_cast<int>(vao);
    memoryStorageService.models.update(modelKey, model);

    for (auto& meshKey: model.meshes)
    {
        auto mesh = memoryStorageService.meshes.get(meshKey);

        for (const auto& resource: mesh.materials)
        {
            auto material = memoryStorageService.materials.get(resource);

            if (material.albedo.has_value() && memoryStorageService.textures.exists(material.albedo.value()))
            {
                auto albedo = memoryStorageService.textures.get(material.albedo.value());
                albedo.gpuResourceId = createTexture(albedo);
                memoryStorageService.textures.update(material.albedo.value(), albedo);
            }

            if (material.normal.has_value() && memoryStorageService.textures.exists(material.normal.value()))
            {
                auto normal = memoryStorageService.textures.get(material.normal.value());
                normal.gpuResourceId = createTexture(normal);
                memoryStorageService.textures.update(material.normal.value(), normal);
            }

            if (material.metallicRoughness.has_value() &&
                memoryStorageService.textures.exists(material.metallicRoughness.value()))
            {
                auto metallicRoughness = memoryStorageService.textures.get(material.metallicRoughness.value());
                metallicRoughness.gpuResourceId = createTexture(metallicRoughness);
                memoryStorageService.textures.update(material.metallicRoughness.value(), metallicRoughness);
            }

            if (material.emissive.has_value() && memoryStorageService.textures.exists(material.emissive.value()))
            {
                auto emissive = memoryStorageService.textures.get(material.emissive.value());
                emissive.gpuResourceId = createTexture(emissive);
                memoryStorageService.textures.update(material.emissive.value(), emissive);
            }

            if (material.occlusion.has_value() && memoryStorageService.textures.exists(material.occlusion.value()))
            {
                auto occlusion = memoryStorageService.textures.get(material.occlusion.value());
                occlusion.gpuResourceId = createTexture(occlusion);
                memoryStorageService.textures.update(material.occlusion.value(), occlusion);
            }

            if (material.environment.has_value())
            {
                auto environment = memoryStorageService.textures.get(material.environment.value());
                environment.gpuResourceId = createTexture(environment);
                memoryStorageService.textures.update(material.environment.value(), environment);
            }

            for (auto& i: material.textures)
            {
                auto texture = memoryStorageService.textures.get(i);
                if (!texture.gpuResourceId.has_value())
                {
                    texture.gpuResourceId = createTexture(texture);
                    memoryStorageService.textures.update(i, texture);
                }
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

unsigned int OpenGLRenderer::createTexture(const Texture& texture) const
{
    unsigned int textureId;

    GLfloat fLargest;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &fLargest);

    glGenTextures(1, &textureId);
    glBindTexture(GL_TEXTURE_2D, textureId);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    auto format = getTextureFormat(texture.format);
    auto type = getTextureDataType(texture.pixelDataType);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture.width, texture.height, NULL, format, type, texture.data.data());

    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_MAX_TEXTURE_MAX_ANISOTROPY, fLargest);

    glBindTexture(GL_TEXTURE_2D, 0);

    return textureId;
}

unsigned int OpenGLRenderer::createCubeMap(
    const Texture& front, const Texture& back, const Texture& left, const Texture& right, const Texture& top,
    const Texture& bottom) const
{
    unsigned int textureId;
    glGenTextures(1, &textureId);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureId);

    const Texture* textures[] = {&right, &left, &bottom, &top, &front, &back};
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
            getTextureFormat(textures[i]->format
            ),
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

unsigned int OpenGLRenderer::createFbo(const Fbo& fbo) const
{
    unsigned int fboGpuResourceId;
    glGenFramebuffers(1, &fboGpuResourceId);
    glBindFramebuffer(GL_FRAMEBUFFER, fboGpuResourceId);

    auto colourAttachmentIndex = 0;
    std::vector<unsigned int> attachments;

    for (auto& attachmentKey: fbo.attachments)
    {
        auto attachment = memoryStorageService.bufferAttachments.get(attachmentKey);

        unsigned int attachmentGpuResourceId;
        glGenTextures(1, &attachmentGpuResourceId);
        glBindTexture(GL_TEXTURE_2D, attachmentGpuResourceId);

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            getTextureFormat(attachment.internalFormat),
            attachment.width,
            attachment.height,
            0,
            getTextureFormat(attachment.captureFormat),
            GL_FLOAT,
            nullptr);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        unsigned int attachmentType = 0;
        switch (attachment.type)
        {
            case FboAttachmentType::Color:
                attachmentType = GL_COLOR_ATTACHMENT0 + colourAttachmentIndex;
                colourAttachmentIndex++;
                break;
            case FboAttachmentType::Depth:
                attachmentType = GL_DEPTH_ATTACHMENT;
                break;
            case FboAttachmentType::Stencil:
                attachmentType = GL_STENCIL_ATTACHMENT;
                break;
            default:
                logger.error("Unknown FBO attachment type {}", attachment.type);
                break;
        }

        glFramebufferTexture2D(GL_FRAMEBUFFER, attachmentType, GL_TEXTURE_2D, attachmentGpuResourceId, 0);
        attachment.gpuResourceId = attachmentGpuResourceId;
        attachments.push_back(attachmentType);

        memoryStorageService.bufferAttachments.update(attachmentKey, attachment);
    }

    glDrawBuffers(static_cast<GLsizei>(attachments.size()), attachments.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return fboGpuResourceId;
}


void OpenGLRenderer::deleteFbo(Fbo& fbo) const
{
    if (fbo.gpuResourceId.has_value())
    {
        glDeleteFramebuffers(1, &fbo.gpuResourceId.value());
    }

    for (auto& attachmentKey: fbo.attachments)
    {
        auto attachment = memoryStorageService.bufferAttachments.get(attachmentKey);
        if (attachment.gpuResourceId.has_value())
        {
            glDeleteTextures(1, &attachment.gpuResourceId.value());
        }
    }
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
    viewportWidth = width;
    viewportHeight = height;
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
    vertices = std::vector<float>({
                                      -1.0, -1.0, 1.0, -1.0,
                                      1.0, 1.0, 1.0, 1.0,
                                      -1.0, 1.0, -1.0, -1.0,
                                      0.0, 0.0, 1.0, 0.0,
                                      1.0, 1.0, 1.0, 1.0,
                                      0.0, 1.0, 0.0, 0.0
                                  });

    glGenVertexArrays(1, &quadVao);
    glBindVertexArray(quadVao);

    GLuint vertexBuffer;
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*) (12 * sizeof(float)));

    glBindVertexArray(0);
}

void OpenGLRenderer::drawFullScreenQuad(const Resource<Shader>& shader,
                                        const std::vector<Resource<FboAttachment>>& attachments) const
{
    glViewport(0, 0, viewportWidth, viewportHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shader->gpuResourceId);

    for (auto i = 0; i < attachments.size(); i++)
    {
        auto attachment = attachments[i];

        if (attachment->gpuResourceId.has_value())
        {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(attachment->gpuResourceId.value()));
        }
    }

    glBindVertexArray(quadVao);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size() / 4));
}

void OpenGLRenderer::drawFullScreenQuad(const Resource<Shader>& shaderKey,
                                        const Resource<FboAttachment>& attachmentKey) const
{
    drawFullScreenQuad(shaderKey, std::vector<Resource<FboAttachment>>{attachmentKey});
}
