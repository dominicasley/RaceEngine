#pragma once

#include <expected>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "../Models/AsyncResult.h"

template<class... Ts>
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
