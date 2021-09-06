#pragma once

#include <vector>
#include <functional>
#include <optional>
#include <future>
#include <iostream>
#include <utility>
#include <spdlog/spdlog.h>
#include "IAsyncTask.h"

template <class T>
class Observable {
private:
    IAsyncTask<T>* task;

    mutable T lastValue;
    mutable std::vector<std::function<void(T)>> subscribers;

public:
    constexpr explicit Observable(IAsyncTask<T>* task = nullptr) : task(task)
    {
    };

    ~Observable() = default;

    void subscribe(const std::function<void(T)>& subscription) const
    {
        subscribers.push_back(subscription);
    }

    void next(T value) const
    {
        lastValue = value;

        for (auto& subscriber : subscribers) {
            subscriber(lastValue);
        }
    }

    T await() const
    {
        if (task)
        {
            return task->await();
        }

        return lastValue;
    }
};
