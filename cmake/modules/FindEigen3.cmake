# Populates target Eigen3::Eigen from official submodule (nightly)
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/eigen-official")
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
set(Eigen3_FOUND true)