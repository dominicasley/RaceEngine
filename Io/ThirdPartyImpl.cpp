// Plain TU on purpose: header-only third-party implementation macros must never
// live inside a module unit.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <tiny_gltf.h>

// clang-19 segfaults emitting tinygltf::TinyGLTF's constructor inside a module
// unit that also imports a PCM whose global module fragment declares it, so the
// loader entry point lives here; GLTFService.cppm declares it extern "C++".
bool raceengineLoadTinyGltf(tinygltf::Model& model, std::string& error, std::string& warning,
                            const std::string& filePath, bool binary)
{
    tinygltf::TinyGLTF gltfLoader;

    return binary ? gltfLoader.LoadBinaryFromFile(&model, &error, &warning, filePath)
                  : gltfLoader.LoadASCIIFromFile(&model, &error, &warning, filePath);
}
