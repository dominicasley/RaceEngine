#pragma once

#include <vector>
#include <memory>
#include <future>
#include <spdlog/logger.h>

#include "../Models/IAsyncTask.h"
#include "../Models/AsyncTask.h"

class BackgroundWorkerService
{
    spdlog::logger& logger;
    static std::vector<std::shared_ptr<ITask>> tasks;

public:
    explicit BackgroundWorkerService(spdlog::logger& logger);

    template <class T>
    static std::shared_ptr<AsyncTask<T>> registerTask(std::future<T> future)
    {
        auto task = std::make_shared<AsyncTask<T>>(std::move(future));
        tasks.emplace_back(task);

        return task;
    }

    void prune();

    void step();
};
