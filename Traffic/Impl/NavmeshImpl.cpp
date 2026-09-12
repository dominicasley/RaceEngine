// Navigation mesh bodies. Declarations are in Api/Navmesh.cppm.
//
// A **module implementation unit** — `module raceengine.traffic;` with no `export` — which is what
// lets it include `<fstream>` and `<chrono>`: it produces an object and no BMI, so nothing imports
// what it includes and nothing rebuilds when it changes.
module;

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <fstream>
#include <ios>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

module raceengine.traffic;

namespace raceengine
{

namespace
{

// --- the scanner --------------------------------------------------------------------------------
//
// A cursor over the document. Every reader below answers false having written `problem`, and the
// top level turns that into the `std::expected` the caller sees, with the byte it stopped at.

struct Cursor
{
    std::string_view text;
    std::size_t at = 0;
    std::string problem{};

    [[nodiscard]] bool done() const
    {
        return at >= text.size();
    }

    [[nodiscard]] char peek() const
    {
        return done() ? '\0' : text[at];
    }

    bool refuse(const std::string_view what)
    {
        problem = std::string(what) + " at byte " + std::to_string(at);

        return false;
    }

    void skipSpace()
    {
        while (!done() && (text[at] == ' ' || text[at] == '\n' || text[at] == '\r' || text[at] == '\t'))
        {
            at++;
        }
    }

    bool expect(const char wanted)
    {
        skipSpace();

        if (peek() != wanted)
        {
            return refuse(std::string("expected '") + wanted + "'");
        }

        at++;

        return true;
    }
};

void appendUtf8(std::string& target, const std::uint32_t point)
{
    if (point < 0x80)
    {
        target.push_back(static_cast<char>(point));
    }
    else if (point < 0x800)
    {
        target.push_back(static_cast<char>(0xC0 | (point >> 6)));
        target.push_back(static_cast<char>(0x80 | (point & 0x3F)));
    }
    else
    {
        target.push_back(static_cast<char>(0xE0 | (point >> 12)));
        target.push_back(static_cast<char>(0x80 | ((point >> 6) & 0x3F)));
        target.push_back(static_cast<char>(0x80 | (point & 0x3F)));
    }
}

bool readNumber(Cursor& cursor, double& value)
{
    cursor.skipSpace();

    const auto* first = cursor.text.data() + cursor.at;
    const auto* last = cursor.text.data() + cursor.text.size();
    const auto answer = std::from_chars(first, last, value);

    if (answer.ec != std::errc{} || answer.ptr == first)
    {
        return cursor.refuse("a number was expected");
    }

    cursor.at = static_cast<std::size_t>(answer.ptr - cursor.text.data());

    return true;
}

// A string, decoded. `keep` is null for a string nobody wants, which is most of them. Keys and area
// names are all this reader keeps, so a `\u` escape outside the basic plane — which no key here
// carries — is written as its two halves rather than paired.
bool readString(Cursor& cursor, std::string* keep)
{
    if (!cursor.expect('"'))
    {
        return false;
    }

    if (keep != nullptr)
    {
        keep->clear();
    }

    while (true)
    {
        if (cursor.done())
        {
            return cursor.refuse("a string does not end");
        }

        const auto character = cursor.text[cursor.at++];

        if (character == '"')
        {
            return true;
        }

        if (character != '\\')
        {
            if (keep != nullptr)
            {
                keep->push_back(character);
            }

            continue;
        }

        if (cursor.done())
        {
            return cursor.refuse("a string does not end");
        }

        const auto escape = cursor.text[cursor.at++];
        auto decoded = '\0';

        switch (escape)
        {
        case '"':
            decoded = '"';
            break;
        case '\\':
            decoded = '\\';
            break;
        case '/':
            decoded = '/';
            break;
        case 'b':
            decoded = '\b';
            break;
        case 'f':
            decoded = '\f';
            break;
        case 'n':
            decoded = '\n';
            break;
        case 'r':
            decoded = '\r';
            break;
        case 't':
            decoded = '\t';
            break;
        case 'u':
        {
            if (cursor.at + 4 > cursor.text.size())
            {
                return cursor.refuse("a \\u escape is cut short");
            }

            auto point = std::uint32_t{0};
            const auto* first = cursor.text.data() + cursor.at;
            const auto answer = std::from_chars(first, first + 4, point, 16);

            if (answer.ec != std::errc{} || answer.ptr != first + 4)
            {
                return cursor.refuse("a \\u escape is not four hex digits");
            }

            cursor.at += 4;

            if (keep != nullptr)
            {
                appendUtf8(*keep, point);
            }

            continue;
        }
        default:
            return cursor.refuse("an escape this reader does not know");
        }

        if (keep != nullptr)
        {
            keep->push_back(decoded);
        }
    }
}

// Every value of an object, by key: `onMember(key)` reads or skips the value that follows.
template <typename OnMember> bool readObject(Cursor& cursor, std::string& key, OnMember onMember)
{
    if (!cursor.expect('{'))
    {
        return false;
    }

    cursor.skipSpace();

    if (cursor.peek() == '}')
    {
        cursor.at++;

        return true;
    }

    while (true)
    {
        if (!readString(cursor, &key) || !cursor.expect(':'))
        {
            return false;
        }

        if (!onMember(std::string_view(key)))
        {
            return false;
        }

        cursor.skipSpace();
        const auto next = cursor.peek();

        if (next == ',')
        {
            cursor.at++;

            continue;
        }

        if (next == '}')
        {
            cursor.at++;

            return true;
        }

        return cursor.refuse("expected ',' or '}'");
    }
}

// Every element of an array: `onElement(index)` reads it.
template <typename OnElement> bool readArray(Cursor& cursor, OnElement onElement)
{
    if (!cursor.expect('['))
    {
        return false;
    }

    cursor.skipSpace();

    if (cursor.peek() == ']')
    {
        cursor.at++;

        return true;
    }

    for (auto index = std::size_t{0};; index++)
    {
        if (!onElement(index))
        {
            return false;
        }

        cursor.skipSpace();
        const auto next = cursor.peek();

        if (next == ',')
        {
            cursor.at++;

            continue;
        }

        if (next == ']')
        {
            cursor.at++;

            return true;
        }

        return cursor.refuse("expected ',' or ']'");
    }
}

bool skipValue(Cursor& cursor);

bool skipValue(Cursor& cursor)
{
    cursor.skipSpace();
    const auto character = cursor.peek();

    if (character == '"')
    {
        return readString(cursor, nullptr);
    }

    if (character == '{')
    {
        auto key = std::string();

        return readObject(cursor, key, [&](const std::string_view) { return skipValue(cursor); });
    }

    if (character == '[')
    {
        return readArray(cursor, [&](const std::size_t) { return skipValue(cursor); });
    }

    for (const auto word : {std::string_view("true"), std::string_view("false"), std::string_view("null")})
    {
        if (cursor.text.substr(cursor.at, word.size()) == word)
        {
            cursor.at += word.size();

            return true;
        }
    }

    auto number = 0.0;

    return readNumber(cursor, number);
}

// One `[n, n, ...]` row of exactly `row.size()` numbers.
bool readRow(Cursor& cursor, const std::span<double> row)
{
    if (!cursor.expect('['))
    {
        return false;
    }

    for (auto index = std::size_t{0}; index < row.size(); index++)
    {
        if (index > 0 && !cursor.expect(','))
        {
            return false;
        }

        if (!readNumber(cursor, row[index]))
        {
            return false;
        }
    }

    return cursor.expect(']');
}

// An array of rows, each handed to `emit(index, row)` as it is read.
template <std::size_t Width, typename Emit> bool readRows(Cursor& cursor, const std::string_view what, Emit emit)
{
    auto row = std::array<double, Width>{};

    return readArray(cursor,
                     [&](const std::size_t index)
                     {
                         if (!readRow(cursor, row))
                         {
                             return cursor.refuse(std::string(what) + " row " + std::to_string(index) + " is not " +
                                                  std::to_string(Width) + " numbers");
                         }

                         return emit(index, row);
                     });
}

// A number that has to be an index: whole, not negative, and inside 32 bits.
[[nodiscard]] bool wholeIndex(const double value, std::uint32_t& index)
{
    if (!(value >= 0.0) || value != std::floor(value) || value >= 4294967295.0)
    {
        return false;
    }

    index = static_cast<std::uint32_t>(value);

    return true;
}

bool readAreas(Cursor& cursor, std::vector<NavMeshArea>& areas)
{
    auto key = std::string();

    return readArray(cursor,
                     [&](const std::size_t)
                     {
                         auto area = NavMeshArea{};
                         auto index = static_cast<std::uint32_t>(areas.size());

                         const auto read =
                             readObject(cursor, key,
                                        [&](const std::string_view member)
                                        {
                                            if (member == "index")
                                            {
                                                auto value = 0.0;

                                                if (!readNumber(cursor, value))
                                                {
                                                    return false;
                                                }

                                                if (!wholeIndex(value, index))
                                                {
                                                    return cursor.refuse("an area index is not a whole number");
                                                }

                                                return true;
                                            }

                                            if (member == "key")
                                            {
                                                return readString(cursor, &area.key);
                                            }

                                            if (member == "cost_hint")
                                            {
                                                return readNumber(cursor, area.costHint);
                                            }

                                            if (member == "friction")
                                            {
                                                return readNumber(cursor, area.friction);
                                            }

                                            return skipValue(cursor);
                                        });

                         if (!read)
                         {
                             return false;
                         }

                         // Eight bits carry the area on every polygon; the bake has four.
                         if (index >= 255)
                         {
                             return cursor.refuse("an area index is out of range");
                         }

                         if (areas.size() <= index)
                         {
                             areas.resize(static_cast<std::size_t>(index) + 1);
                         }

                         areas[index] = std::move(area);

                         return true;
                     });
}

bool readPolygons(Cursor& cursor, std::vector<NavMeshPolygon>& polygons)
{
    return readRows<10>(cursor, "polygon",
                        [&](const std::size_t index, const std::array<double, 10>& row)
                        {
                            auto polygon = NavMeshPolygon{.x0 = row[0],
                                                          .z0 = row[1],
                                                          .x1 = row[2],
                                                          .z1 = row[3],
                                                          .y00 = row[4],
                                                          .y01 = row[5],
                                                          .y11 = row[6],
                                                          .y10 = row[7]};

                            if (!wholeIndex(row[8], polygon.area) || !wholeIndex(row[9], polygon.component))
                            {
                                return cursor.refuse("polygon row " + std::to_string(index) +
                                                     " carries an area or a component that is not a whole number");
                            }

                            polygons.push_back(polygon);

                            return true;
                        });
}

bool readLinks(Cursor& cursor, std::vector<NavMeshLink>& links)
{
    return readRows<8>(cursor, "link",
                       [&](const std::size_t index, const std::array<double, 8>& row)
                       {
                           auto link = NavMeshLink{.p = glm::dvec3(row[2], row[3], row[4]),
                                                   .q = glm::dvec3(row[5], row[6], row[7])};

                           if (!wholeIndex(row[0], link.a) || !wholeIndex(row[1], link.b))
                           {
                               return cursor.refuse("link row " + std::to_string(index) +
                                                    " names a polygon by something that is not a whole number");
                           }

                           links.push_back(link);

                           return true;
                       });
}

// --- geometry -----------------------------------------------------------------------------------

[[nodiscard]] double bilinear(const NavMesh& mesh, const std::uint32_t polygon, const double x, const double z)
{
    const auto width = mesh.x1[polygon] - mesh.x0[polygon];
    const auto depth = mesh.z1[polygon] - mesh.z0[polygon];
    const auto u = width > 0.0 ? std::clamp((x - mesh.x0[polygon]) / width, 0.0, 1.0) : 0.0;
    const auto v = depth > 0.0 ? std::clamp((z - mesh.z0[polygon]) / depth, 0.0, 1.0) : 0.0;

    return (1.0 - u) * (1.0 - v) * mesh.y00[polygon] + (1.0 - u) * v * mesh.y01[polygon] + u * v * mesh.y11[polygon] +
           u * (1.0 - v) * mesh.y10[polygon];
}

[[nodiscard]] int cellOf(const double coordinate, const double origin, const double cell, const int count)
{
    return std::clamp(static_cast<int>(std::floor((coordinate - origin) / cell)), 0, count - 1);
}

} // namespace

std::expected<NavMeshSource, std::string> parseNavMesh(const std::string_view document)
{
    auto cursor = Cursor{.text = document};
    auto source = NavMeshSource{};
    auto key = std::string();
    auto inner = std::string();

    const auto read = readObject(
        cursor, key,
        [&](const std::string_view member)
        {
            if (member == "areas")
            {
                return readAreas(cursor, source.areas);
            }

            if (member == "polygons")
            {
                source.polygons.reserve(source.statedPolygons);

                return readPolygons(cursor, source.polygons);
            }

            if (member == "links")
            {
                source.links.reserve(source.statedLinks);

                return readLinks(cursor, source.links);
            }

            if (member == "counts")
            {
                return readObject(cursor, inner,
                                  [&](const std::string_view count)
                                  {
                                      if (count == "polygons" || count == "links")
                                      {
                                          auto value = 0.0;
                                          auto whole = std::uint32_t{0};

                                          if (!readNumber(cursor, value))
                                          {
                                              return false;
                                          }

                                          if (!wholeIndex(value, whole))
                                          {
                                              return cursor.refuse("a count is not a whole number");
                                          }

                                          (count == "polygons" ? source.statedPolygons : source.statedLinks) = whole;

                                          return true;
                                      }

                                      return skipValue(cursor);
                                  });
            }

            if (member == "agent")
            {
                return readObject(cursor, inner,
                                  [&](const std::string_view field)
                                  {
                                      if (field == "radius_m")
                                      {
                                          return readNumber(cursor, source.agentRadiusMetres);
                                      }

                                      return skipValue(cursor);
                                  });
            }

            return skipValue(cursor);
        });

    if (!read)
    {
        return std::unexpected("the navmesh document is not readable: " + cursor.problem);
    }

    cursor.skipSpace();

    if (!cursor.done())
    {
        return std::unexpected("the navmesh document carries content after its root object");
    }

    if (source.polygons.empty())
    {
        return std::unexpected("the navmesh document carries no polygons");
    }

    // The document's own summary against what was read, the traffic export's rule: a mismatch is an
    // export that lost rows on the way out, and a mesh with a hole in it routes a unit into a wall.
    if (source.statedPolygons != 0 && source.statedPolygons != source.polygons.size())
    {
        return std::unexpected("the navmesh document claims " + std::to_string(source.statedPolygons) +
                               " polygons and carries " + std::to_string(source.polygons.size()));
    }

    if (source.statedLinks != 0 && source.statedLinks != source.links.size())
    {
        return std::unexpected("the navmesh document claims " + std::to_string(source.statedLinks) +
                               " links and carries " + std::to_string(source.links.size()));
    }

    return source;
}

std::expected<NavMesh, std::string> buildNavMesh(const NavMeshSource& source, const NavMeshOptions& options)
{
    if (source.areas.empty())
    {
        return std::unexpected("the navmesh states no areas");
    }

    if (source.polygons.empty())
    {
        return std::unexpected("the navmesh carries no polygons");
    }

    if (!(options.bucketCellMetres > 0.0))
    {
        return std::unexpected("the navmesh bucket cell must be positive");
    }

    auto mesh = NavMesh{};
    mesh.areas = source.areas;
    mesh.agentRadiusMetres = source.agentRadiusMetres;
    mesh.cellMetres = options.bucketCellMetres;

    // --- the polygons of the one component, and the map from the document's index to the kept one --
    auto remap = std::vector<std::uint32_t>(source.polygons.size(), noPolygon);
    auto componentsSeen = std::vector<bool>();

    for (auto index = std::size_t{0}; index < source.polygons.size(); index++)
    {
        const auto& polygon = source.polygons[index];

        if (polygon.component != options.component)
        {
            mesh.droppedPolygons++;

            if (componentsSeen.size() <= polygon.component)
            {
                componentsSeen.resize(static_cast<std::size_t>(polygon.component) + 1, false);
            }

            if (!componentsSeen[polygon.component])
            {
                componentsSeen[polygon.component] = true;
                mesh.droppedComponents++;
            }

            continue;
        }

        if (!(polygon.x1 > polygon.x0) || !(polygon.z1 > polygon.z0))
        {
            return std::unexpected("navmesh polygon " + std::to_string(index) + " is not a rectangle");
        }

        if (polygon.area >= source.areas.size())
        {
            return std::unexpected("navmesh polygon " + std::to_string(index) + " names area " +
                                   std::to_string(polygon.area) + " of " + std::to_string(source.areas.size()));
        }

        for (const auto height : {polygon.y00, polygon.y01, polygon.y11, polygon.y10})
        {
            if (!std::isfinite(height))
            {
                return std::unexpected("navmesh polygon " + std::to_string(index) +
                                       " carries a height that is not finite");
            }
        }

        remap[index] = static_cast<std::uint32_t>(mesh.x0.size());

        mesh.x0.push_back(polygon.x0);
        mesh.z0.push_back(polygon.z0);
        mesh.x1.push_back(polygon.x1);
        mesh.z1.push_back(polygon.z1);
        mesh.y00.push_back(polygon.y00);
        mesh.y01.push_back(polygon.y01);
        mesh.y11.push_back(polygon.y11);
        mesh.y10.push_back(polygon.y10);
        mesh.area.push_back(static_cast<std::uint8_t>(polygon.area));
    }

    if (mesh.x0.empty())
    {
        return std::unexpected("the navmesh has no polygon in component " + std::to_string(options.component));
    }

    // --- the links between kept polygons ---------------------------------------------------------
    for (auto index = std::size_t{0}; index < source.links.size(); index++)
    {
        const auto& link = source.links[index];

        if (link.a >= source.polygons.size() || link.b >= source.polygons.size())
        {
            return std::unexpected("navmesh link " + std::to_string(index) +
                                   " names a polygon the document does not carry");
        }

        if (link.a == link.b)
        {
            return std::unexpected("navmesh link " + std::to_string(index) + " joins a polygon to itself");
        }

        const auto a = remap[link.a];
        const auto b = remap[link.b];

        if (a == noPolygon || b == noPolygon)
        {
            mesh.droppedLinks++;

            continue;
        }

        mesh.linkA.push_back(a);
        mesh.linkB.push_back(b);
        mesh.portalP.push_back(link.p);
        mesh.portalQ.push_back(link.q);
    }

    // --- adjacency: each link counted at both ends, then laid out in compressed rows -----------------
    const auto polygons = mesh.x0.size();
    mesh.neighbourStarts.assign(polygons + 1, 0);

    for (auto index = std::size_t{0}; index < mesh.linkA.size(); index++)
    {
        mesh.neighbourStarts[mesh.linkA[index] + 1]++;
        mesh.neighbourStarts[mesh.linkB[index] + 1]++;
    }

    for (auto index = std::size_t{0}; index < polygons; index++)
    {
        mesh.neighbourStarts[index + 1] += mesh.neighbourStarts[index];
    }

    mesh.neighbourLinks.assign(mesh.neighbourStarts.back(), 0);

    {
        auto fill = std::vector<std::uint32_t>(mesh.neighbourStarts.begin(), mesh.neighbourStarts.end() - 1);

        for (auto index = std::size_t{0}; index < mesh.linkA.size(); index++)
        {
            mesh.neighbourLinks[fill[mesh.linkA[index]]++] = static_cast<std::uint32_t>(index);
            mesh.neighbourLinks[fill[mesh.linkB[index]]++] = static_cast<std::uint32_t>(index);
        }
    }

    // --- the bucket --------------------------------------------------------------------------------
    auto minX = std::numeric_limits<double>::max();
    auto minZ = std::numeric_limits<double>::max();
    auto maxX = std::numeric_limits<double>::lowest();
    auto maxZ = std::numeric_limits<double>::lowest();

    for (auto index = std::size_t{0}; index < polygons; index++)
    {
        minX = std::min(minX, mesh.x0[index]);
        minZ = std::min(minZ, mesh.z0[index]);
        maxX = std::max(maxX, mesh.x1[index]);
        maxZ = std::max(maxZ, mesh.z1[index]);
    }

    mesh.originX = minX;
    mesh.originZ = minZ;
    mesh.columns = std::max(1, static_cast<int>(std::ceil((maxX - minX) / mesh.cellMetres)));
    mesh.rows = std::max(1, static_cast<int>(std::ceil((maxZ - minZ) / mesh.cellMetres)));

    const auto cells = static_cast<std::size_t>(mesh.columns) * static_cast<std::size_t>(mesh.rows);
    mesh.bucketStarts.assign(cells + 1, 0);

    const auto span = [&](const std::uint32_t polygon, int& c0, int& c1, int& r0, int& r1)
    {
        c0 = cellOf(mesh.x0[polygon], mesh.originX, mesh.cellMetres, mesh.columns);
        c1 = cellOf(mesh.x1[polygon], mesh.originX, mesh.cellMetres, mesh.columns);
        r0 = cellOf(mesh.z0[polygon], mesh.originZ, mesh.cellMetres, mesh.rows);
        r1 = cellOf(mesh.z1[polygon], mesh.originZ, mesh.cellMetres, mesh.rows);
    };

    for (auto polygon = std::uint32_t{0}; polygon < polygons; polygon++)
    {
        auto c0 = 0;
        auto c1 = 0;
        auto r0 = 0;
        auto r1 = 0;
        span(polygon, c0, c1, r0, r1);

        for (auto row = r0; row <= r1; row++)
        {
            for (auto column = c0; column <= c1; column++)
            {
                mesh.bucketStarts[static_cast<std::size_t>(row) * static_cast<std::size_t>(mesh.columns) +
                                  static_cast<std::size_t>(column) + 1]++;
            }
        }
    }

    for (auto index = std::size_t{0}; index < cells; index++)
    {
        mesh.bucketStarts[index + 1] += mesh.bucketStarts[index];
    }

    mesh.bucketEntries.assign(mesh.bucketStarts.back(), 0);

    {
        auto fill = std::vector<std::uint32_t>(mesh.bucketStarts.begin(), mesh.bucketStarts.end() - 1);

        for (auto polygon = std::uint32_t{0}; polygon < polygons; polygon++)
        {
            auto c0 = 0;
            auto c1 = 0;
            auto r0 = 0;
            auto r1 = 0;
            span(polygon, c0, c1, r0, r1);

            for (auto row = r0; row <= r1; row++)
            {
                for (auto column = c0; column <= c1; column++)
                {
                    const auto cell = static_cast<std::size_t>(row) * static_cast<std::size_t>(mesh.columns) +
                                      static_cast<std::size_t>(column);
                    mesh.bucketEntries[fill[cell]++] = polygon;
                }
            }
        }
    }

    return mesh;
}

std::expected<NavMesh, std::string> loadNavMesh(const std::string& filePath, const NavMeshOptions& options)
{
    using Clock = std::chrono::steady_clock;
    const auto seconds = [](const Clock::time_point from, const Clock::time_point to)
    {
        return std::chrono::duration<double>(to - from).count();
    };

    const auto started = Clock::now();

    auto stream = std::ifstream(filePath, std::ios::binary | std::ios::ate);
    if (!stream.is_open())
    {
        return std::unexpected("Unable to open navmesh with path " + filePath);
    }

    const auto size = stream.tellg();
    if (size < 0)
    {
        return std::unexpected("Unable to size navmesh " + filePath);
    }

    auto document = std::string(static_cast<std::size_t>(size), '\0');
    stream.seekg(0, std::ios::beg);

    if (!stream.read(document.data(), static_cast<std::streamsize>(size)))
    {
        return std::unexpected("Unable to read navmesh " + filePath);
    }

    const auto read = Clock::now();

    auto source = parseNavMesh(document);
    if (!source)
    {
        return std::unexpected("Navmesh " + filePath + ": " + source.error());
    }

    // The 33 MB of text is not needed past this point, and the build below allocates about as much.
    document = std::string();

    const auto parsed = Clock::now();

    auto mesh = buildNavMesh(source.value(), options);
    if (!mesh)
    {
        return std::unexpected("Navmesh " + filePath + ": " + mesh.error());
    }

    const auto built = Clock::now();

    mesh->timing = NavMeshTiming{.readSeconds = seconds(started, read),
                                 .parseSeconds = seconds(read, parsed),
                                 .buildSeconds = seconds(parsed, built)};

    return mesh;
}

double navMeshHeight(const NavMesh& mesh, const std::uint32_t polygon, const double x, const double z)
{
    return bilinear(mesh, polygon, x, z);
}

glm::dvec3 navMeshCentre(const NavMesh& mesh, const std::uint32_t polygon)
{
    const auto x = 0.5 * (mesh.x0[polygon] + mesh.x1[polygon]);
    const auto z = 0.5 * (mesh.z0[polygon] + mesh.z1[polygon]);

    return glm::dvec3(x, bilinear(mesh, polygon, x, z), z);
}

std::uint32_t navMeshNeighbour(const NavMesh& mesh, const std::uint32_t polygon, const std::uint32_t link)
{
    return mesh.linkA[link] == polygon ? mesh.linkB[link] : mesh.linkA[link];
}

double navMeshPlanarDistance(const NavMesh& mesh, const std::uint32_t polygon, const double x, const double z)
{
    const auto dx = std::max({mesh.x0[polygon] - x, 0.0, x - mesh.x1[polygon]});
    const auto dz = std::max({mesh.z0[polygon] - z, 0.0, z - mesh.z1[polygon]});

    return std::hypot(dx, dz);
}

NavMeshPlace locateOnNavMesh(const NavMesh& mesh, const glm::dvec3& pointMetres, const double reachMetres,
                             const double heightToleranceMetres)
{
    auto best = NavMeshPlace{};

    if (mesh.columns <= 0 || mesh.rows <= 0 || mesh.x0.empty())
    {
        return best;
    }

    const auto reach = std::max(reachMetres, 0.0);
    const auto x = pointMetres.x;
    const auto z = pointMetres.z;

    // The cells the reach can touch, or none where the point is off the bucket by more than it.
    const auto lowColumn = std::floor((x - reach - mesh.originX) / mesh.cellMetres);
    const auto highColumn = std::floor((x + reach - mesh.originX) / mesh.cellMetres);
    const auto lowRow = std::floor((z - reach - mesh.originZ) / mesh.cellMetres);
    const auto highRow = std::floor((z + reach - mesh.originZ) / mesh.cellMetres);

    if (highColumn < 0.0 || highRow < 0.0 || lowColumn >= static_cast<double>(mesh.columns) ||
        lowRow >= static_cast<double>(mesh.rows))
    {
        return best;
    }

    const auto c0 = std::max(0, static_cast<int>(lowColumn));
    const auto c1 = std::min(mesh.columns - 1, static_cast<int>(highColumn));
    const auto r0 = std::max(0, static_cast<int>(lowRow));
    const auto r1 = std::min(mesh.rows - 1, static_cast<int>(highRow));

    auto bestScore = std::numeric_limits<double>::max();

    for (auto row = r0; row <= r1; row++)
    {
        for (auto column = c0; column <= c1; column++)
        {
            const auto cell = static_cast<std::size_t>(row) * static_cast<std::size_t>(mesh.columns) +
                              static_cast<std::size_t>(column);

            for (auto entry = mesh.bucketStarts[cell]; entry < mesh.bucketStarts[cell + 1]; entry++)
            {
                const auto polygon = mesh.bucketEntries[entry];
                const auto planar = navMeshPlanarDistance(mesh, polygon, x, z);

                if (planar > reach)
                {
                    continue;
                }

                const auto height = bilinear(mesh, polygon, x, z);
                const auto vertical = std::abs(pointMetres.y - height);

                if (vertical > heightToleranceMetres)
                {
                    continue;
                }

                const auto score = std::hypot(planar, vertical);

                if (score < bestScore)
                {
                    bestScore = score;
                    best = NavMeshPlace{
                        .polygon = polygon, .heightMetres = height, .planarMetres = planar, .verticalMetres = vertical};
                }
            }
        }
    }

    return best;
}

} // namespace raceengine
