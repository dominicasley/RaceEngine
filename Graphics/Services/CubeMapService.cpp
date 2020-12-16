
#include "CubeMapService.h"

CubeMapService::CubeMapService(
    spdlog::logger& logger,
    OpenGLRenderer& renderer,
    MemoryStorageService& memoryStorageService) :
    logger(logger),
    renderer(renderer),
    memoryStorageService(memoryStorageService)
{

}

CubeMap* CubeMapService::create(const std::string& name, Texture* front, Texture* back, Texture* left,
                                       Texture* right, Texture* top, Texture* bottom) const
{
    auto gpuResourceId = renderer.createCubeMap(front, back, left, right, top, bottom);

    return memoryStorageService.cubeMaps.add(name, CubeMap {
        .gpuResourceId = gpuResourceId,
        .front = front,
        .back = back,
        .left = left,
        .right = right,
        .top = top,
        .bottom = bottom
    });
}
