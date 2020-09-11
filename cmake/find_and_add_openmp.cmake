macro(find_and_add_openmp TARGET)
  set(_QUIET ${ARGN})

  find_package(OpenMP ${_QUIET})

  # Have to check several OPENMP_FOUND due to bug in
  # one version of CMake and the docs (fixed in patch release)
  # OpenMP is missing on macOS llvm default, for example
  if(OpenMP_FOUND OR OPENMP_FOUND OR OpenMP_CXX_FOUND)

    # target_link_libraries(${TARGET} INTERFACE OpenMP::OpenMP_CXX) # does not work on interface, so we also must use set_property
    # CMake 3.9 FindOpenMP allows correct linking with Clang in more cases
    find_package(Threads REQUIRED)
    # Clang may need -fopenmp=libiomp5 instead, can't be detected here without CMake 3.9
    set_property(TARGET ${TARGET}
      APPEND PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
    set_property(TARGET ${TARGET}
      APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
  else()
    # just guessing flag for clang
    if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
      set(OpenMP_C_FLAG "-fopenmp=libomp -Wno-unused-command-line-argument")
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      set(OpenMP_CXX_FLAG "-fopenmp=libomp -Wno-unused-command-line-argument")
    endif()
  endif() # OpenMP_FOUND

endmacro()
