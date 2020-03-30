cmake_minimum_required (VERSION 3.8)

add_library (Async
        Models/IAsyncTask.h
        Models/Observable.h
        Services/BackgroundWorkerService.cpp
        Services/BackgroundWorkerService.h
        Operators/ForkJoin.h Models/AsyncTask.h Models/ITask.h)

target_include_directories(Async PUBLIC "../../Middleware/")
target_link_directories(Async PRIVATE "../../Middleware/spdlog/lib")
target_link_libraries(Async PRIVATE spdlog)
