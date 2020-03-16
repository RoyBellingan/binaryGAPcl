cmake_minimum_required(VERSION 3.16)

project(p1 LANGUAGES CXX)

#set (CMAKE_CXX_FLAGS "-std=c++20")
#set (CMAKE_CXX_FLAGS "-O3 -g -march=native -flto -fuse-ld=gold -fuse-linker-plugin ")
#set (CMAKE_EXE_LINKER_FLAGS "-flto")
set (CMAKE_CXX_FLAGS "-O0 -g -march=native")
set (CMAKE_EXE_LINKER_FLAGS "-lOpenCL")
set (CMAKE_VERBOSE_MAKEFILE "true")
set (CMAKE_CXX_COMPILER "/usr/bin/g++-10")
add_executable(p1 main.cpp)
