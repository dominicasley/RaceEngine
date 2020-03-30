#pragma once

#include <future>
#include <memory>
#include "IAsyncTask.h"
#include "Observable.h"

template<class T>
class AsyncTask : public IAsyncTask<T>
{
private:
    T value;
    std::future<T> future;
    Observable<T> observer;
    std::mutex futureMutex;

public:
    explicit AsyncTask(std::future<T> future)  : future(std::move(future)), observer(this)
    {
    }

    void step() override
    {
        std::lock_guard<std::mutex> futureLock(futureMutex);

        if (future.valid() && future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
            value = future.get();
            complete = true;
            observer.next(value);
        }
    }

    T await() override
    {
        std::lock_guard<std::mutex> futureLock(futureMutex);

        if (future.valid()) {
            value = future.get();
            complete = true;
            observer.next(value);
        }

        return value;
    }

    constexpr const Observable<T>& asObservable() const
    {
        return observer;
    }
};
