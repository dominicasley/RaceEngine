#include "BackgroundWorkerService.h"

#include <algorithm>

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
