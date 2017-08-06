cmake_minimum_required(VERSION 3.6)
project(PhaMers)

include_directories("src")
include_directories("include")

add_compile_options(-std=c++11)
SET( CMAKE_CXX_FLAGS  "-g -Wall -pedantic -O0 -std=c++0x -D_GLIBCXX_USE_NANOSLEEP -pthread")

set(SRC src/thread-pool.cc src/semaphore.cc src/ostreamlock.cc src/thread-utils.cc)

add_library(threadpool STATIC ${SRC})
add_library(kmer-counter lib/)