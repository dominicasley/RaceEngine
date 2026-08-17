#pragma once

#include <expected>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <spdlog/logger.h>

#include "../Models/AsyncResult.h"

class BackgroundWorkerService
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
