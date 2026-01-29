cmake_minimum_required(VERSION 3.10)
project(CSV_Reader VERSION 1.0)

add_library(csv INTERFACE)
target_include_directories(csv INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
