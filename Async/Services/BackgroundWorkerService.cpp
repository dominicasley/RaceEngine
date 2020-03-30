#include "BackgroundWorkerService.h"
#include <iostream>

std::vector<std::shared_ptr<ITask>> BackgroundWorkerService::tasks;

BackgroundWorkerService::BackgroundWorkerService(spdlog::logger& logger) : logger(logger)
{
}

void BackgroundWorkerService::step()
{
    BackgroundWorkerService::prune();

    for (const auto& task : tasks)
    {
         task->step();
    }
}

void BackgroundWorkerService::prune()
{
    tasks.erase(
        std::remove_if(
            tasks.begin(),
            tasks.end(),
            [](auto& e) { return e->complete; }),
        tasks.end());
}
