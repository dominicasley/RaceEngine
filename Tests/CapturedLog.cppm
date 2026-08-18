module;

#include <memory>
#include <sstream>
#include <string>

#include <spdlog/logger.h>
#include <spdlog/sinks/ostream_sink.h>

export module raceengine.tests.log;

namespace raceengine::tests
{

// A logger nothing else in the process can reach: constructed directly rather than through
// spdlog's registry, so a test neither registers a global name nor collides with one. Engine
// still registers "engine" through spdlog::stdout_color_mt, which is why two Engines cannot
// coexist — nothing here builds an Engine, and this is what makes that unnecessary.
//
// The engine's own report-once behaviour is only observable through what reaches a sink, so the
// sink is the assertion surface: the pattern is stripped to the message alone.
export class CapturedLog
{
public:
    CapturedLog() :
        logger("raceengine-test", std::make_shared<spdlog::sinks::ostream_sink_mt>(stream))
    {
        logger.set_pattern("%v");
        logger.set_level(spdlog::level::trace);
    }

    CapturedLog(const CapturedLog&) = delete;
    CapturedLog(CapturedLog&&) = delete;
    CapturedLog& operator=(const CapturedLog&) = delete;
    CapturedLog& operator=(CapturedLog&&) = delete;
    ~CapturedLog() = default;

    [[nodiscard]] spdlog::logger& sink()
    {
        return logger;
    }

    [[nodiscard]] std::string text()
    {
        logger.flush();

        return stream.str();
    }

    [[nodiscard]] size_t occurrences(const std::string& needle)
    {
        const auto haystack = text();
        size_t found = 0;

        for (auto at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + 1))
        {
            found++;
        }

        return found;
    }

private:
    // Declared before the logger: the sink is built from it in the member initialiser list, and
    // that list runs in declaration order.
    std::ostringstream stream;
    spdlog::logger logger;
};

} // namespace raceengine::tests
