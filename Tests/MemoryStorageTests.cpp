#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

import raceengine;

using raceengine::MemoryStorage;
using raceengine::Resource;

namespace
{

// A payload with an interior heap allocation, because the invariant that matters most about
// mutate() is what happens to that interior: update() used to copy-assign a whole element into
// the slot, which destroyed and replaced every allocation inside it while the slot address
// survived.
struct Payload
{
    std::string name;
    int value = 0;
};

} // namespace

TEST_CASE("a default-constructed handle names nothing", "[storage]")
{
    MemoryStorage<Payload> storage;

    const auto unissued = Resource<Payload>{};

    REQUIRE_FALSE(unissued.issued());
    REQUIRE_FALSE(storage.exists(unissued));
    REQUIRE(storage.find(unissued) == nullptr);
    REQUIRE_THROWS_AS(storage.get(unissued), std::out_of_range);

    // Generation 0 is never issued, so the default handle stays unmatched even once the storage
    // holds the element it would otherwise have named by index. This is the case the old
    // {0, nullptr} handle got wrong: a bounds check passed and the null pointer was dereferenced.
    const auto issued = storage.add(Payload{.name = "first", .value = 1});

    REQUIRE(issued.index == unissued.index);
    REQUIRE_FALSE(storage.exists(unissued));
    REQUIRE(storage.exists(issued));
}

TEST_CASE("remove retires every handle to the slot", "[storage]")
{
    MemoryStorage<Payload> storage;

    const auto handle = storage.add(Payload{.name = "doomed", .value = 7});
    const auto copyOfHandle = handle;

    REQUIRE(storage.size() == 1);
    REQUIRE(storage.remove(handle));
    REQUIRE(storage.size() == 0);

    REQUIRE_FALSE(storage.exists(handle));
    REQUIRE_FALSE(storage.exists(copyOfHandle));
    REQUIRE(storage.find(handle) == nullptr);
    REQUIRE_THROWS_AS(storage.get(handle), std::out_of_range);

    // The second removal is a no-op rather than a double free: two materials of one model naming
    // the same texture is exactly this, and it has to be safe.
    REQUIRE_FALSE(storage.remove(handle));
}

TEST_CASE("a recycled slot issues a new generation and rejects the stale handle", "[storage]")
{
    MemoryStorage<Payload> storage;

    const auto stale = storage.add(Payload{.name = "first", .value = 1});
    REQUIRE(storage.remove(stale));

    const auto recycled = storage.add(Payload{.name = "second", .value = 2});

    // Same slot, later generation: this is the whole of the lifetime model.
    REQUIRE(recycled.index == stale.index);
    REQUIRE(recycled.generation > stale.generation);

    REQUIRE_FALSE(storage.exists(stale));
    REQUIRE(storage.exists(recycled));
    REQUIRE(storage.find(stale) == nullptr);
    REQUIRE(storage.get(recycled).name == "second");
}

TEST_CASE("the free list reuses slots instead of growing the deque", "[storage]")
{
    MemoryStorage<Payload> storage;

    std::vector<Resource<Payload>> firstRound;
    for (auto index = 0; index < 4; index++)
    {
        firstRound.push_back(storage.add(Payload{.name = "round one", .value = index}));
    }

    for (const auto& handle : firstRound)
    {
        REQUIRE(storage.remove(handle));
    }

    REQUIRE(storage.size() == 0);

    // A level that loads, unloads and reloads has to settle at a bounded slot count rather than
    // climbing, so every index of the second round must come out of the first round's range.
    for (auto index = 0; index < 4; index++)
    {
        const auto handle = storage.add(Payload{.name = "round two", .value = index});

        REQUIRE(handle.index < 4);
        REQUIRE(storage.exists(handle));
    }

    REQUIRE(storage.size() == 4);
}

TEST_CASE("mutate writes the element in place", "[storage]")
{
    MemoryStorage<Payload> storage;

    const auto handle = storage.add(Payload{.name = "before", .value = 1});
    const auto* borrowBefore = storage.find(handle);
    REQUIRE(borrowBefore != nullptr);

    REQUIRE(storage.mutate(handle,
                           [](Payload& target)
                           {
                               target.name = "after";
                               target.value = 2;
                           }));

    // The same object, not a replacement: a borrow taken before the mutation still names it and
    // sees what the mutator wrote. Nothing in the API can assign a whole element into a slot any
    // more, which is what makes that safe to say.
    REQUIRE(storage.find(handle) == borrowBefore);
    REQUIRE(borrowBefore->name == "after");
    REQUIRE(borrowBefore->value == 2);
}

TEST_CASE("mutate on a stale handle reports and does not run the mutator", "[storage]")
{
    MemoryStorage<Payload> storage;

    const auto stale = storage.add(Payload{.name = "gone", .value = 1});
    REQUIRE(storage.remove(stale));

    auto mutatorRan = false;
    REQUIRE_FALSE(storage.mutate(stale, [&](Payload&) { mutatorRan = true; }));
    REQUIRE_FALSE(mutatorRan);
}

TEST_CASE("a borrow survives a concurrent-style add", "[storage]")
{
    MemoryStorage<Payload> storage;

    const auto handle = storage.add(Payload{.name = "kept", .value = 1});
    const auto* borrow = storage.find(handle);
    REQUIRE(borrow != nullptr);

    // add() runs on background workers while the main thread reads through a borrow it took at
    // the top of a loop nest. The deque backing is what makes that sound, and this is the only
    // place that says so in code rather than in a comment.
    for (auto index = 0; index < 64; index++)
    {
        const auto appended = storage.add(Payload{.name = "appended", .value = index});
        REQUIRE(appended.index != handle.index);
    }

    REQUIRE(storage.find(handle) == borrow);
    REQUIRE(borrow->name == "kept");
}

TEST_CASE("find reports a stale handle where get treats it as a caller bug", "[storage]")
{
    MemoryStorage<Payload> storage;

    const auto live = storage.add(Payload{.name = "live", .value = 1});
    const auto stale = storage.add(Payload{.name = "stale", .value = 2});
    REQUIRE(storage.remove(stale));

    REQUIRE(storage.find(live) != nullptr);
    REQUIRE(storage.find(stale) == nullptr);

    REQUIRE_NOTHROW(storage.get(live));
    REQUIRE_THROWS_AS(storage.get(stale), std::out_of_range);
}

TEST_CASE("size counts live elements rather than slots", "[storage]")
{
    MemoryStorage<Payload> storage;

    const auto first = storage.add(Payload{.name = "a", .value = 1});
    const auto second = storage.add(Payload{.name = "b", .value = 2});
    const auto third = storage.add(Payload{.name = "c", .value = 3});

    REQUIRE(storage.size() == 3);
    REQUIRE(storage.remove(second));
    REQUIRE(storage.size() == 2);

    REQUIRE(storage.exists(first));
    REQUIRE(storage.exists(third));
}
