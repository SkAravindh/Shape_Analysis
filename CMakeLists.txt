cmake_minimum_required(VERSION 3.22)
project(ShapeAnalysis VERSION 1.0 LANGUAGES CXX)

# Define C++ standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 0N)

#Define libraries and executable location(i.e Static, Shared, Executable)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "CMake build options are: Release, Debug" FORCE)
endif()
message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")

find_package(Eigen3 REQUIRED)

# Add subdirectories
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Dependencies/geometry-central)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Dependencies/polyscope)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Dependencies/LBFGSpp)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Dependencies/spectra)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Dependencies/CSV_Reader)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Src)

