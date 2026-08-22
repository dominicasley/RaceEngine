module;

#include <expected>
#include <ranges>
#include <string>
#include <vector>

export module raceengine.graphics:FboService;

import :IGpuResourceFactory;
import raceengine.graphics.models;
import raceengine.shared;

namespace raceengine
{

export class FboService
{
private:
    MemoryStorageService& memoryStorageService;
    IGpuResourceFactory& gpuResourceFactory;

public:
    explicit FboService(MemoryStorageService& memoryStorageService, IGpuResourceFactory& gpuResourceFactory);
    [[nodiscard]] std::expected<Resource<Fbo>, std::string> create(const CreateFboDTO& createFboDTO) const;
    [[nodiscard]] std::expected<void, std::string> recreate(const Resource<Fbo>& fbo) const;
    [[nodiscard]] std::expected<void, std::string> resize(const Resource<Fbo>& fbo, unsigned int width,
                                                          unsigned int height) const;

    [[nodiscard]] std::vector<Resource<FboAttachment>> getAttachmentsOfType(const Fbo& fbo,
                                                                            FboAttachmentType type) const;
};

// A framebuffer that fails to come back is left with no GPU id rather than the stale one the
// delete just invalidated: whoever handles the error is choosing what to do about a buffer
// that cannot be drawn into, not about one that can still be drawn into by accident.
//
// The framebuffer is copied out rather than mutated in place because deleteFbo/createFbo take it
// by reference and reach into the attachment storage themselves; only the resulting id is written
// back through the element, and it is written through mutate(), which touches that one field.

} // namespace raceengine
