export module raceengine.resource;

namespace raceengine
{

// A handle into MemoryStorage<T>: the slot it names, plus the generation that slot carried when
// the handle was issued. There is no cached pointer any more. The pointer was what made "storage
// is add-only" the lifetime model — a raw address into a deque can only stay valid while nothing
// is ever destroyed — and it is what a handle has to stop carrying before anything can be freed.
//
// Generation 0 is never issued, so a default-constructed handle names nothing in any storage.
// That matters: the old `{0, nullptr}` passed the bounds check exists() used to be, for every
// storage that held at least one element, and then dereferenced null.
//
// Trivially copyable and eight bytes: handles are copied freely, into scene elements and into
// std::optional slots on the model types, and none of that copies anything but the two numbers.
export template <typename T> struct Resource
{
    unsigned int index{};
    unsigned int generation{};

    // Whether this handle was ever issued by a storage. It says nothing about the element still
    // being there — only MemoryStorage<T>::exists can say that, because only the storage holds
    // the generation to compare against.
    [[nodiscard]] constexpr bool issued() const
    {
        return generation != 0;
    }

    [[nodiscard]] constexpr bool operator==(const Resource& other) const = default;
};

} // namespace raceengine
