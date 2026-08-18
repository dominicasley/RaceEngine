# ozz-animation 0.17.0 declares cmake_minimum_required(VERSION 3.30) but builds
# fine with 3.28 (this machine's CMake); relax the declaration after fetch.
file(READ "CMakeLists.txt" content)
string(REPLACE "cmake_minimum_required(VERSION 3.30)"
               "cmake_minimum_required(VERSION 3.28)" content "${content}")
file(WRITE "CMakeLists.txt" "${content}")
