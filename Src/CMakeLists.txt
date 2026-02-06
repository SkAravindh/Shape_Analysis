
set(Header FunctionalMapping.h
        MeshProcessor.h
        Descriptors.h
        Utils.h
        FMoptimizer.h
        FunctionalMapEnergyEvaluator.h
)

set(Source FunctionalMapping.cpp
        MeshProcessor.cpp
        Descriptors.cpp
        Utils.cpp
        FMoptimizer.cpp
        FunctionalMapEnergyEvaluator.cpp
)


# Create a library named Functional from the source and header files (if needed).
add_library(Functional_Map STATIC ${Header} ${Source})

add_library(LBFGSpp INTERFACE)
target_include_directories(LBFGSpp INTERFACE ${PROJECT_SOURCE_DIR}/Dependencies/LBFGSpp/include)

#add_library(spectra INTERFACE)
#target_include_directories(spectra INTERFACE ${PROJECT_SOURCE_DIR}/Dependencies/spectra/include)

add_library(CSV_Reader INTERFACE)
target_include_directories(CSV_Reader INTERFACE ${PROJECT_SOURCE_DIR}/Dependencies/CSV_Reader/include)

# Link all dependencies required by the Functional library’s headers and source files.
target_link_libraries(Functional_Map PUBLIC geometry-central Eigen3::Eigen polyscope LBFGSpp CSV_Reader)
#target_link_libraries(Functional_Map PUBLIC geometry-central Eigen3::Eigen polyscope LBFGSpp spectra CSV_Reader)

# Specify all necessary include directories.
#target_include_directories(Functional_Map PUBLIC ${PROJECT_SOURCE_DIR}/Dependencies/LBFGSpp/include
#                                                ${PROJECT_SOURCE_DIR}/Dependencies/spectra/include)

add_executable(shape_analysis main.cpp)
target_link_libraries(shape_analysis PRIVATE Functional_Map)