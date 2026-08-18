export module raceengine.resource;

namespace raceengine
{

export template <typename T> class Resource
{
public:
    unsigned long long id;
    const T* value;

    const T* operator->() const
    {
        return value;
    }
};

} // namespace raceengine
