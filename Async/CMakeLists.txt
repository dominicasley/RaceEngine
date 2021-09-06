set(CMAKE_CXX_STANDARD 20)
cmake_minimum_required (VERSION 3.8)

add_library (Async
        Models/IAsyncTask.h
        Models/Observable.h
        Services/BackgroundWorkerService.cpp
        Services/BackgroundWorkerService.h
        Operators/ForkJoin.h Models/AsyncTask.h Models/ITask.h)

target_link_libraries(Async PRIVATE spdlog)
