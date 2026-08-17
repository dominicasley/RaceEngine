#pragma once

#include <expected>
#include <future>
#include <string>

template<typename T>
using AsyncResult = std::future<std::expected<T, std::string>>;
