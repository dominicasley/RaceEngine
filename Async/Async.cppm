module;

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <expected>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <spdlog/logger.h>

export module raceengine.async;

namespace raceengine
{

export template<typename T>
using AsyncResult = std::future<std::expected<T, std::string>>;

export template<class... Ts>
[[nodiscard]] std::expected<std::tuple<Ts...>, std::string> awaitAll(AsyncResult<Ts>... pending)
{
    auto results = std::make_tuple(pending.get()...);

    auto firstError = std::optional<std::string>();
    std::apply(
        [&firstError](auto&... result) {
            ([&] {
                if (!firstError && !result)
                {
                    firstError = std::move(result).error();
                }
            }(), ...);
        },
        results);

    if (firstError)
    {
        return std::unexpected(std::move(*firstError));
    }

    return std::apply(
        [](auto&... result) { return std::tuple<Ts...>(std::move(result).value()...); },
        results);
}

export class BackgroundWorkerService
{
    template<class T>
    struct UnwrapExpected
    {
        using type = T;
    };

    template<class U>
    struct UnwrapExpected<std::expected<U, std::string>>
    {
        using type = U;
    };

    spdlog::logger& logger;
    std::mutex mutex;
    std::condition_variable_any wake;
    std::deque<std::move_only_function<void()>> queue;
    // workers last: they stop and join before the queue/mutex above are destroyed.
    std::vector<std::jthread> workers;

    void pump(const std::stop_token& stopToken);

public:
    explicit BackgroundWorkerService(spdlog::logger& logger);
    ~BackgroundWorkerService();

    template<class F>
    [[nodiscard]] auto submit(F work) -> AsyncResult<typename UnwrapExpected<std::invoke_result_t<F>>::type>
    {
        using T = typename UnwrapExpected<std::invoke_result_t<F>>::type;

        auto promise = std::promise<std::expected<T, std::string>>();
        auto result = promise.get_future();

        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.emplace_back([promise = std::move(promise), work = std::move(work)]() mutable {
                try
                {
                    promise.set_value(work());
                }
                catch (const std::exception& e)
                {
                    promise.set_value(std::unexpected(e.what()));
                }
                catch (...)
                {
                    promise.set_value(std::unexpected("unknown error"));
                }
            });
        }

        wake.notify_one();

        return result;
    }
};

} // namespace raceengine

module :private;

namespace raceengine
{

BackgroundWorkerService::BackgroundWorkerService(spdlog::logger& logger) : logger(logger)
{
    const auto workerCount = std::max(1u, std::min(4u, std::thread::hardware_concurrency()));

    workers.reserve(workerCount);
    for (auto i = 0u; i < workerCount; i++)
    {
        workers.emplace_back([this](const std::stop_token& stopToken) { pump(stopToken); });
    }
}

BackgroundWorkerService::~BackgroundWorkerService()
{
    for (auto& worker : workers)
    {
        worker.request_stop();
    }
}

void BackgroundWorkerService::pump(const std::stop_token& stopToken)
{
    while (true)
    {
        auto work = std::move_only_function<void()>();

        {
            std::unique_lock<std::mutex> lock(mutex);

            if (!wake.wait(lock, stopToken, [this] { return !queue.empty(); }))
            {
                return;
            }

            work = std::move(queue.front());
            queue.pop_front();
        }

        work();
    }
}

} // namespace raceengine
