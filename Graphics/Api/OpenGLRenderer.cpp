#include "OpenGLRenderer.h"
#include <gl/gl3w.h>
#include <vector>
#include "Resources/RenderableEntityDesc.h"
#include "Resources/Mesh.h"
#include "Resources/Shader.h"
#include "../Models/Scene/Scene.h"
#include "../Models/Scene/RenderableEntity.h"
#include "../Models/Scene/Light.h"
#include "../Models/Scene/Camera.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <tiny_gltf.h>

OpenGLRenderer::OpenGLRenderer(
    spdlog::logger& logger,
    SceneService& sceneService,
    RenderableEntityService& renderableEntityService,
    CameraService& cameraService) :
    logger(logger),
    sceneService(sceneService),
    renderableEntityService(renderableEntityService),
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

void OpenGLRenderer::drawMesh(tinygltf::Model* model, tinygltf::Mesh& mesh)
{
    for (const auto& primitive : mesh.primitives)
    {
        const auto material = model->materials[primitive.material];

        if (model->textures.size() > material.pbrMetallicRoughness.baseColorTexture.index)
        {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(
                GL_TEXTURE_2D,
                static_cast<GLuint>(model->textures[material.pbrMetallicRoughness.baseColorTexture.index].extras.GetNumberAsInt()));
        }

        if (model->textures.size() > material.normalTexture.index)
        {
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(
                GL_TEXTURE_2D,
                static_cast<GLuint>(model->textures[material.normalTexture.index].extras.GetNumberAsInt()));
        }

        if (model->textures.size() > material.pbrMetallicRoughness.metallicRoughnessTexture.index)
        {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(
                GL_TEXTURE_2D,
                static_cast<GLuint>(model->textures[material.pbrMetallicRoughness.metallicRoughnessTexture.index].extras.GetNumberAsInt()));
        }

        if (model->textures.size() > material.emissiveTexture.index)
        {
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(
                GL_TEXTURE_2D,
                static_cast<GLuint>(model->textures[material.emissiveTexture.index].extras.GetNumberAsInt()));
        }

        if (model->textures.size() > material.occlusionTexture.index)
        {
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(
                GL_TEXTURE_2D,
                static_cast<GLuint>(model->textures[material.occlusionTexture.index].extras.GetNumberAsInt()));
        }

        tinygltf::Accessor indexAccessor = model->accessors[primitive.indices];

        glDrawElements(primitive.mode,
                       static_cast<GLsizei>(indexAccessor.count),
                       indexAccessor.componentType,
                       (void*) (indexAccessor.byteOffset));
    }
}

void OpenGLRenderer::drawModelNodes(RenderableEntity& entity, tinygltf::Model* model, tinygltf::Node& node)
{
    if (node.mesh != -1)
    {
        const auto joints = renderableEntityService.joints(entity, model, node);
        setProgramUniform(1, "jointTransformationMatrixes", joints);
        setProgramUniform(1, "animated", !joints.empty());

        drawMesh(model, model->meshes[node.mesh]);
    }

    for (int i : node.children)
    {
        drawModelNodes(entity, model, model->nodes[i]);
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
        const auto entityModelMatrix = renderableEntityService.modelMatrix(*entity);


        setProgramUniform(1, "localToScreen4x4Matrix", camera->modelViewProjectionMatrix * entityModelMatrix);
        setProgramUniform(1, "localToWorld4x4Matrix", entityModelMatrix);
        setProgramUniform(1, "localToWorld3x3Matrix", glm::mat3(entityModelMatrix));


        setProgramUniform(1, "normalMatrix",
                          glm::transpose(glm::inverse(glm::mat3(camera->modelViewMatrix * entityModelMatrix))));

        auto model = entity->model;
        glBindVertexArray(static_cast<GLuint>(model->extras.GetNumberAsInt()));

        const auto& currentScene = model->scenes[model->defaultScene];

        for (int node : currentScene.nodes)
        {
            drawModelNodes(*entity, model, model->nodes[node]);
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

std::vector<unsigned int>& OpenGLRenderer::bindMesh(
    std::vector<unsigned int>& vbos,
    tinygltf::Model* model,
    tinygltf::Mesh& mesh)
{
    for (const auto& bufferView : model->bufferViews)
    {
        if (bufferView.target == 0)
        {
            logger.warn("bufferView.target is zero. skipping buffer view.");
            continue;
        }

        tinygltf::Buffer buffer = model->buffers[bufferView.buffer];

        GLuint vbo;
        glGenBuffers(1, &vbo);
        vbos.push_back(vbo);

        glBindBuffer(bufferView.target, vbo);
        glBufferData(bufferView.target, bufferView.byteLength,
                     buffer.data.data() + bufferView.byteOffset, GL_STATIC_DRAW);
    }

    for (const auto& primitive : mesh.primitives)
    {
        for (auto& attrib : primitive.attributes)
        {
            auto accessor = model->accessors[attrib.second];
            auto byteStride = accessor.ByteStride(model->bufferViews[accessor.bufferView]);

            glBindBuffer(GL_ARRAY_BUFFER, vbos[accessor.bufferView]);

            int size = 1;
            if (accessor.type != TINYGLTF_TYPE_SCALAR)
            {
                size = accessor.type;
            }

            int vertexAttribute = -1;
            if (attrib.first == "POSITION")
            {
                vertexAttribute = 0;
            }
            else if (attrib.first == "TEXCOORD_0")
            {
                vertexAttribute = 1;
            }
            else if (attrib.first == "NORMAL")
            {
                vertexAttribute = 2;
            }
            else if (attrib.first == "JOINTS_0")
            {
                vertexAttribute = 3;
            }
            else if (attrib.first == "WEIGHTS_0")
            {
                vertexAttribute = 4;
            }

            if (vertexAttribute > -1)
            {
                glEnableVertexAttribArray(vertexAttribute);
                glVertexAttribPointer(vertexAttribute, size, accessor.componentType,
                                      accessor.normalized ? GL_TRUE : GL_FALSE,
                                      byteStride, (void*) (accessor.byteOffset));
            }
        }
    }

    for (auto& texture : model->textures)
    {
        tinygltf::Image& image = model->images[texture.source];
        auto textureId = static_cast<int>(createTexture(texture, image));
        texture.extras = tinygltf::Value(textureId);
    }

    return vbos;
}

void OpenGLRenderer::bindModelNodes(std::vector<unsigned int>& vbos, tinygltf::Model* model, tinygltf::Node& node)
{
    if (node.mesh != -1)
    {
        bindMesh(vbos, model, model->meshes[node.mesh]);
    }

    for (auto i : node.children)
    {
        bindModelNodes(vbos, model, model->nodes[i]);
    }
}

void OpenGLRenderer::upload(tinygltf::Model* model)
{
    GLuint vao;
    std::vector<unsigned int> vbos;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    const tinygltf::Scene& scene = model->scenes[model->defaultScene];

    for (auto node : scene.nodes)
    {
        bindModelNodes(vbos, model, model->nodes[node]);
    }

    glBindVertexArray(0);

    for (unsigned int& vbo : vbos)
    {
        glDeleteBuffers(1, &vbo);
    }

    model->extras = tinygltf::Value(static_cast<int>(vao));
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

unsigned int OpenGLRenderer::createTexture(const tinygltf::Texture& texture, tinygltf::Image& image)
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
        image.component == 1 ? GL_RED : image.component == 1 ? GL_RG : image.component == 3 ? GL_RGB : GL_RGBA;
    GLenum type = image.bits == 16 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE;

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, NULL, format, type, image.image.data());
    glBindTexture(GL_TEXTURE_2D, 0);

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
