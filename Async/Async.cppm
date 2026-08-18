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

export module raceengine.async;

namespace raceengine
{

export template <typename T> using AsyncResult = std::future<std::expected<T, std::string>>;

export template <class... Ts>
[[nodiscard]] std::expected<std::tuple<Ts...>, std::string> awaitAll(AsyncResult<Ts>... pending)
{
    auto results = std::make_tuple(pending.get()...);

    auto firstError = std::optional<std::string>();
    std::apply(
        [&firstError](auto&... result)
        {
            (
                [&]
                {
                    if (!firstError && !result)
                    {
                        firstError = std::move(result).error();
                    }
                }(),
                ...);
        },
        results);

    if (firstError)
    {
        return std::unexpected(std::move(*firstError));
    }

    return std::apply([](auto&... result) { return std::tuple<Ts...>(std::move(result).value()...); }, results);
}

export class BackgroundWorkerService
{
    template <class T> struct UnwrapExpected
    {
        using type = T;
    };

    template <class U> struct UnwrapExpected<std::expected<U, std::string>>
    {
        using type = U;
    };

    // A queued job runs its work, or — once shutdown has begun — is cancelled: it resolves its
    // future with an error and never touches the services the work captured. The cancel flag is
    // part of the job because the promise is the only thing that survives type erasure.
    using Job = std::move_only_function<void(bool cancelled)>;

    std::mutex mutex;
    std::condition_variable_any wake;
    std::deque<Job> queue;
    // workers last: they stop and join before the queue/mutex above are destroyed.
    std::vector<std::jthread> workers;

    void pump(const std::stop_token& stopToken);

public:
    BackgroundWorkerService();
    ~BackgroundWorkerService();

    template <class F>
    [[nodiscard]] auto submit(F work) -> AsyncResult<typename UnwrapExpected<std::invoke_result_t<F>>::type>
    {
        using T = typename UnwrapExpected<std::invoke_result_t<F>>::type;

        auto promise = std::promise<std::expected<T, std::string>>();
        auto result = promise.get_future();

        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.emplace_back(
                [promise = std::move(promise), work = std::move(work)](bool cancelled) mutable
                {
                    if (cancelled)
                    {
                        promise.set_value(std::unexpected("cancelled: background workers are shutting down"));
                        return;
                    }

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

BackgroundWorkerService::BackgroundWorkerService()
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

    // Jobs still queued are cancelled here rather than left to the queue's destructor: their
    // futures resolve with an error instead of a broken promise, and the work itself — which
    // captured services this pool outlives — never runs. Workers have already stopped taking
    // from the queue by the time stop is observed, so nothing else can claim these.
    auto pending = std::deque<Job>();
    {
        std::lock_guard<std::mutex> lock(mutex);
        pending.swap(queue);
    }

    for (auto& job : pending)
    {
        job(true);
    }
}

void BackgroundWorkerService::pump(const std::stop_token& stopToken)
{
    while (true)
    {
        auto job = Job();

        {
            std::unique_lock<std::mutex> lock(mutex);

            // wait returns the predicate, so a stop request with a non-empty queue returns
            // true: without this check the pool would drain on shutdown and run jobs against
            // services already destroyed. Pending work is discarded, not executed.
            if (!wake.wait(lock, stopToken, [this] { return !queue.empty(); }) || stopToken.stop_requested())
            {
                return;
            }

            job = std::move(queue.front());
            queue.pop_front();
        }

        job(false);
    }
}

} // namespace raceengine
