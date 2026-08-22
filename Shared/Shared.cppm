module;

#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ozz/animation/runtime/animation.h>
#include <ozz/animation/runtime/skeleton.h>

export module raceengine.shared;

import raceengine.graphics.models;
import raceengine.resource;

namespace raceengine
{

// Throws std::runtime_error carrying `message`. It exists for a build-time reason rather than a
// design one, and the reason is specific enough to be worth stating: `#include <stdexcept>` in a
// global module fragment costs about nine seconds in any translation unit that also imports
// `raceengine`, because every declaration that header states has to be merged against the copy
// already sitting inside the imported BMIs. Measured 2026-08-22 on clang-19, one sandbox unit:
// 1.7s without the include, 10.8s with it, everything else held. It is `<stdexcept>` specifically
// and not a rule about global module fragments — <vector> costs nothing the same way, <string>
// three tenths of a second, <exception> and <new> about as little.
//
// So a game constructor that has to fail calls this instead of naming the exception, and its
// global module fragment gets that much shorter. The exception type and what propagates out of it
// are unchanged; only the throw site moves in here.
//
// Declared here and defined below the private fragment, so an importer never deserializes the body
// either.
//
// A constructor is the one place in this project that still throws: everything able to report a
// failure to a caller returns std::expected instead, and that rule is not relaxed by this.
export [[noreturn]] void fail(const std::string& message);

// Generational slot storage. std::deque still backs it and elements still never move, but that
// address stability is no longer doing four jobs: it is only what makes a resolved borrow
// (find/get below) usable for the length of a loop nest, while a background thread appends.
//
// **Lifetime** is the generation. remove() destroys the element and frees its slot; the next
// add() to reuse that slot bumps the generation, so every handle issued against the old element
// fails the identity check instead of reading whatever refilled the slot. exists() is that
// identity check, not a bounds check.
//
// **Ownership** is the caller's: this storage owns the element's memory and nothing else. A GPU
// id inside an element is released by whoever removes the element (AssetService), through
// IGpuResourceFactory, *before* the element goes.
//
// **Concurrency.** add() runs on background workers while the main thread reads; mutate() and
// remove() are the main thread's alone. The mutex covers the deque's bookkeeping, the free list
// and every slot write, so add-vs-read, add-vs-add and mutate-vs-add are sound by construction.
// One thing is sound by discipline and is written down here because nothing in the type says it:
// get()/find() hand back a reference that outlives the lock. It stays valid until that element is
// removed, and removal happens on the main thread between frames — so a borrow taken and used
// inside one frame cannot be pulled out from under itself. A background thread must not hold one
// across a frame boundary.
export template <typename T> class MemoryStorage
{
    struct Slot
    {
        std::optional<T> value{};
        // The generation this slot is currently issuing against. Zero only before the slot has
        // ever been filled, which is what makes Resource<T>{} match no live slot anywhere.
        unsigned int generation = 0;
    };

    mutable std::mutex accessorMutex;
    std::deque<Slot> slots;
    std::vector<unsigned int> freeSlots;

    // Both require the lock. The const overload is what every read path goes through, so the
    // whole identity check is three predicates on words that sit next to the payload — cheaper
    // than the lock around it, which is why it is never compiled out.
    [[nodiscard]] Slot* liveSlot(const Resource<T>& key)
    {
        if (!key.issued() || key.index >= slots.size())
        {
            return nullptr;
        }

        auto& slot = slots[key.index];

        return slot.generation == key.generation && slot.value.has_value() ? &slot : nullptr;
    }

    [[nodiscard]] const Slot* liveSlot(const Resource<T>& key) const
    {
        if (!key.issued() || key.index >= slots.size())
        {
            return nullptr;
        }

        const auto& slot = slots[key.index];

        return slot.generation == key.generation && slot.value.has_value() ? &slot : nullptr;
    }

    // Requires the lock. A freed slot is reused before the deque grows, so a level that loads,
    // unloads and reloads settles at a bounded slot count rather than climbing.
    [[nodiscard]] unsigned int claimSlot()
    {
        if (!freeSlots.empty())
        {
            const auto index = freeSlots.back();
            freeSlots.pop_back();

            return index;
        }

        slots.emplace_back();

        return static_cast<unsigned int>(slots.size() - 1);
    }

public:
    MemoryStorage() = default;
    MemoryStorage(const MemoryStorage&) = delete;
    MemoryStorage(MemoryStorage&&) = delete;
    MemoryStorage& operator=(const MemoryStorage&) = delete;
    MemoryStorage& operator=(MemoryStorage&&) = delete;
    ~MemoryStorage() = default;

    [[nodiscard]] Resource<T> add(const T& item)
    {
        std::lock_guard<std::mutex> lock(accessorMutex);

        const auto index = claimSlot();
        auto& slot = slots[index];
        slot.value.emplace(item);
        slot.generation++;

        return Resource<T>{.index = index, .generation = slot.generation};
    }

    [[nodiscard]] Resource<T> add(T&& item)
    {
        std::lock_guard<std::mutex> lock(accessorMutex);

        const auto index = claimSlot();
        auto& slot = slots[index];
        slot.value.emplace(std::move(item));
        slot.generation++;

        return Resource<T>{.index = index, .generation = slot.generation};
    }

    // Identity, not bounds: true only while the element this exact handle was issued for is
    // still there.
    [[nodiscard]] bool exists(const Resource<T>& key) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);

        return liveSlot(key) != nullptr;
    }

    // The element, or nullptr if the handle no longer names one. This is the resolved borrow the
    // draw path uses: resolve once at the outermost loop where the handle is constant and read
    // fields through it, instead of paying a lookup — and a lock — per field the way
    // `resource->field` would now have to. The borrow is valid until that element is removed;
    // see the threading note on the class.
    [[nodiscard]] const T* find(const Resource<T>& key) const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        const auto* slot = liveSlot(key);

        return slot == nullptr ? nullptr : &slot->value.value();
    }

    // The same borrow for callers that have already established the handle is live. A stale or
    // out-of-range handle here is a caller bug rather than a runtime condition, so it is reported
    // as one — this used to index the deque unchecked and hand back whatever was there.
    [[nodiscard]] const T& get(const Resource<T>& key) const
    {
        const auto* value = find(key);
        if (value == nullptr)
        {
            throw std::out_of_range("Resource handle " + std::to_string(key.index) + "/" +
                                    std::to_string(key.generation) + " names no live element in this storage");
        }

        return *value;
    }

    // In-place mutation of the element, under the lock. This replaced update(), and the
    // replacement is the point: update() copy-assigned a whole element into the slot, so every
    // heap allocation *inside* it was destroyed and replaced while the slot address survived. An
    // iterator into Mesh::meshPrimitives, or a reference to Mesh::name, taken before the call was
    // dangling after it — the sharpest rule in the codebase, obeyed only by accident of ordering
    // and written down nowhere. There is no whole-element assignment in this API any more, so
    // there is nothing to invalidate: a borrow taken before a mutate() names the same object
    // after it, with the fields the mutator wrote.
    //
    // The mutator runs while this storage's mutex is held, so it must not call back into the same
    // MemoryStorage. Other storages have their own mutexes and nest freely; the engine nests
    // models -> meshes and models -> materials -> textures, in that order.
    //
    // False means the handle was stale and nothing ran.
    template <typename Mutator> bool mutate(const Resource<T>& key, Mutator&& mutator)
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        auto* slot = liveSlot(key);
        if (slot == nullptr)
        {
            return false;
        }

        std::forward<Mutator>(mutator)(slot->value.value());

        return true;
    }

    // Destroys the element and retires every handle to its slot. False means the handle was
    // already stale, which is what makes releasing a texture two materials of one model share a
    // second no-op rather than a double free.
    //
    // Whatever the element owned outside this storage — a GPU id — has to be released before
    // this, because this is what makes the id unreachable.
    bool remove(const Resource<T>& key)
    {
        std::lock_guard<std::mutex> lock(accessorMutex);
        auto* slot = liveSlot(key);
        if (slot == nullptr)
        {
            return false;
        }

        slot->value.reset();
        freeSlots.push_back(key.index);

        return true;
    }

    // Live elements, not slots: a slot on the free list holds nothing and is not counted.
    [[nodiscard]] size_t size() const
    {
        std::lock_guard<std::mutex> lock(accessorMutex);

        return slots.size() - freeSlots.size();
    }
};

export class MemoryStorageService
{
public:
    MemoryStorage<Model> models;
    MemoryStorage<Mesh> meshes;
    MemoryStorage<Texture> textures;
    MemoryStorage<Material> materials;
    MemoryStorage<CubeMap> cubeMaps;
    MemoryStorage<Shader> shaders;
    MemoryStorage<Fbo> frameBuffers;
    MemoryStorage<FboAttachment> bufferAttachments;
    MemoryStorage<PostProcess> postProcesses;
    MemoryStorage<std::unique_ptr<ozz::animation::Skeleton>> skeletons;
    MemoryStorage<std::unique_ptr<ozz::animation::Animation>> animations;
};

} // namespace raceengine

module :private;

namespace raceengine
{

void fail(const std::string& message)
{
    throw std::runtime_error(message);
}

} // namespace raceengine
