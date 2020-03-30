#pragma once

#include <Logger.h>
#include <vector>
#include <memory>
#include <future>

#include "../Models/IAsyncTask.h"
#include "../Models/AsyncTask.h"

class BackgroundWorkerService
{
    Logger& logger;
    static std::vector<std::shared_ptr<ITask>> tasks;

public:
    BackgroundWorkerService(Logger& logger);

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
