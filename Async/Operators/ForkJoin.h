#pragma once

#include <tuple>
#include <iostream>
#include <functional>

#include "../Models/Observable.h"

class ForkJoin
{
public:
    ForkJoin() = delete;

    template<class... Ts>
    static const Observable<std::tuple<Ts...>>& joinAsync(const Observable<Ts>& ... observables) {
        return BackgroundWorkerService::registerTask(
            std::async(std::launch::async, &ForkJoin::join<Ts...>, observables...)
        )->asObservable();
    }

    template<class... Ts>
    constexpr static std::tuple<Ts...> join(const Observable<Ts>& ... observables) {
        return std::make_tuple(observables.await()...);
    }
};