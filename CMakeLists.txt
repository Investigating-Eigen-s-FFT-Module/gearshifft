cmake_minimum_required(VERSION 3.7) # better language support, more features
# https://cliutils.gitlab.io/modern-cmake/chapters/intro/newcmake.html
# cmake configurations are inspired by VexCL
# See: https://github.com/ddemidov/vexcl/

if(${CMAKE_VERSION} VERSION_LESS 3.12)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.12)
endif()
#-------------------------------------------------------------------------------
if(NOT GEARSHIFFT_ROOT)
  set(GEARSHIFFT_ROOT "${CMAKE_SOURCE_DIR}")
endif()


# cmake module path with additional finders
list(APPEND CMAKE_MODULE_PATH
  ${GEARSHIFFT_ROOT}/cmake
  ${GEARSHIFFT_ROOT}/cmake/modules
  )
#-------------------------------------------------------------------------------
# guard against in-source builds and default build-type
include(safeguards)
include(init_build_type)
#-------------------------------------------------------------------------------

# gearshifft cmake options (requires GEARSHIFFT_ROOT)
include(gearshifft_options)

# Superbuild
option(GEARSHIFFT_USE_SUPERBUILD "Invokes gearshifft superbuild" OFF)

if(GEARSHIFFT_USE_SUPERBUILD)
  project(GEARSHIFFT_SUPERBUILD C CXX) # .. NONE would overwrite CMAKE_CXX_COMPILER flag

  # execute the superbuild (this script will be invoked again without the
  # USE_SUPERBUILD option this time)
  include(superbuild)

  return()                      # <-- STOP processing this file further

endif()

#-------------------------------------------------------------------------------

project(gearshifft
  VERSION 0.4.1
  LANGUAGES C CXX)

# half-code for half precision data type
if(GEARSHIFFT_FLOAT16_SUPPORT)
  include(ext/half)
endif()

include(find_and_add_openmp)

#------------------------------------------------------------------------------

if(GEARSHIFFT_VERBOSE)
  set(_QUIET "")
else()
  set(_QUIET "QUIET")
endif()

#------------------------------------------------------------------------------
# Boost
#------------------------------------------------------------------------------

if(GEARSHIFFT_USE_STATIC_LIBS)
  set(Boost_USE_STATIC_LIBS ON) # otherwise find libs automatically
  message(STATUS "Static linking with Boost system libraries may give linker problems. Enable GEARSHIFFT_SUPERBUILD_EXT_DOWNLOAD_Boost then (requires GEARSHIFFT_USE_SUPERBUILD).")
endif()
find_package(Boost 1.59 CONFIG REQUIRED ${_QUIET} COMPONENTS
  system
  unit_test_framework
  program_options
  )

include(boost_targets)

#----------------------------------------------------------------------------
# Generic target
#----------------------------------------------------------------------------

add_library(Common INTERFACE)
add_library(gearshifft::Common ALIAS Common)

# setting language requirements for all targets
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# TODO: in cmake 3.8+ use target_compile_features(myTarget PUBLIC cxx_std_11)

target_include_directories(Common INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/inc>
#  $<INSTALL_INTERFACE:include> # include dirs are not installed
#  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> # include dirs are not installed
  )

# do not export this interface (only for internal usage requirements)
target_compile_options(Common INTERFACE
  # g++
  $<$<CXX_COMPILER_ID:GNU>:$<BUILD_INTERFACE:-Wall>>
  $<$<CXX_COMPILER_ID:GNU>:$<BUILD_INTERFACE:-Wextra>>
  $<$<CXX_COMPILER_ID:GNU>:$<BUILD_INTERFACE:-Wold-style-cast>>
#  $<$<CXX_COMPILER_ID:GNU>:$<BUILD_INTERFACE:-Wshadow>> # maybe too many false positives
  # Clang
  $<$<CXX_COMPILER_ID:Clang>:$<BUILD_INTERFACE:-Wall>>
  $<$<CXX_COMPILER_ID:Clang>:$<BUILD_INTERFACE:-Wextra>>
  $<$<CXX_COMPILER_ID:Clang>:$<BUILD_INTERFACE:-Wshadow>> # see above, does better job than gcc
  # MSVC
  $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
  $<$<CXX_COMPILER_ID:MSVC>:/wd4003>
  $<$<CXX_COMPILER_ID:MSVC>:/wd4996>
  )

# gearshifft version.
# As this information is built as own library, targets only have to relink,
# when version changes.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gearshifft_version.cpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/gearshifft_version.cpp"
  @ONLY)

add_library(gearshifft_version STATIC
  "${CMAKE_CURRENT_BINARY_DIR}/gearshifft_version.cpp")

target_link_libraries(gearshifft_version PUBLIC Common)

# continue with Common interface used by all gearshifft targets
target_compile_definitions(Common INTERFACE
  $<$<CXX_COMPILER_ID:GNU>:_GLIBCXX_USE_CXX11_ABI=${GEARSHIFFT_CXX11_ABI}>
  $<BUILD_INTERFACE:GEARSHIFFT_INSTALL_CONFIG_FILE=${CMAKE_SOURCE_DIR}/share/gearshifft/extents.conf>
  $<INSTALL_INTERFACE:GEARSHIFFT_INSTALL_CONFIG_FILE=${GEARSHIFFT_INSTALL_CONFIG_PATH}/extents.conf>
  GEARSHIFFT_FLOAT16_SUPPORT=${GEARSHIFFT_FLOAT16_SUPPORT}
  GEARSHIFFT_PRECISION_HALF_ONLY=${GEARSHIFFT_PRECISION_HALF_ONLY}
  GEARSHIFFT_PRECISION_SINGLE_ONLY=${GEARSHIFFT_PRECISION_SINGLE_ONLY}
  GEARSHIFFT_PRECISION_DOUBLE_ONLY=${GEARSHIFFT_PRECISION_DOUBLE_ONLY}
  GEARSHIFFT_DUMP_FREQUENCY=${GEARSHIFFT_DUMP_FREQUENCY}

  BOOST_ALL_NO_LIB=1
  )
#TODO: check paths

# TODO: Is there no generator expression for string matching?
if (${CMAKE_CXX_COMPILER} MATCHES ".*scorep.*")
    target_compile_definitions(Common INTERFACE GEARSHIFFT_SCOREP_INSTRUMENTATION)
endif()

if (GEARSHIFFT_FLUSH_CACHE)
  target_compile_definitions(Common INTERFACE
    GEARSHIFFT_FLUSH_CACHE
    GEARSHIFFT_FLUSH_LLC_SIZE_MEBIBYTES=${GEARSHIFFT_FLUSH_LLC_SIZE_MEBIBYTES}
    GEARSHIFFT_FLUSH_CL_SIZE_BYTES=${GEARSHIFFT_FLUSH_CL_SIZE_BYTES}
  )
  message(STATUS "Cache flush between runs enabled")
endif()

if(NOT Boost_VERSION VERSION_LESS 1.68.0)
  # https://github.com/boostorg/system/issues/26
  target_compile_definitions(Common INTERFACE BOOST_ERROR_CODE_HEADER_ONLY)
endif()

target_link_libraries(Common INTERFACE
  gearshifft_version
  Boost::system
  Boost::unit_test_framework
  Boost::program_options
  Boost::thread
  ${CMAKE_DL_LIBS}
  )

if(GEARSHIFFT_FLOAT16_SUPPORT)
  target_link_libraries(Common INTERFACE half)
endif()
#------------------------------------------------------------------------------
# Find gearshifft back-ends
#------------------------------------------------------------------------------

# CUDA+CUFFT
if(GEARSHIFFT_BACKEND_CUFFT)
  find_package(CUDA CONFIG ${_QUIET}) # no imported targets, will be language in cmake >= 3.10
  if(CUDA_FOUND)

    add_library(CUFFT INTERFACE)
    add_library(gearshifft::CUFFT ALIAS CUFFT)

    target_include_directories(CUFFT INTERFACE ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(CUFFT INTERFACE CUFFT_ENABLED)
    target_link_libraries(CUFFT INTERFACE
      Common
      ${CUDA_LIBRARIES})

    if(GEARSHIFFT_USE_STATIC_LIBS AND CUDA_VERSION VERSION_LESS 9.2)
      message(WARNING "CUDA <9.2 and static cufft build not supported. Using shared linking for cufft.")
    endif()
    if(GEARSHIFFT_USE_STATIC_LIBS AND NOT CUDA_VERSION VERSION_LESS 9.2)
      if(WIN32)
        message(FATAL_ERROR "CuFFT does not support static linking under Windows.")
      endif()

      # Static library without callback support (to avoid separate device compile step with nvcc)
      find_library(CUDA_CUFFT_STATIC_LIBRARY
        NAMES cufft_static_nocallback
        PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        NO_DEFAULT_PATH)
      if(NOT CUDA_CUFFT_STATIC_LIBRARY)
        message(FATAL_ERROR "Could not find static cufft library.")
      endif()

      find_library(CUDA_CULIBOS_LIBRARY
        NAMES culibos
        PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        NO_DEFAULT_PATH)
      if(NOT CUDA_CULIBOS_LIBRARY)
        message(FATAL_ERROR "Could not find culibos library.")
      endif()
      target_link_libraries(CUFFT INTERFACE ${CUDA_CUFFT_STATIC_LIBRARY} ${CUDA_CULIBOS_LIBRARY})
    else()
      target_link_libraries(CUFFT INTERFACE ${CUDA_CUFFT_LIBRARIES})
    endif()

    message(STATUS " gearshifft::CUFFT enabled.")

  endif()
endif()

#------------------------------------------------------------------------------
# OPENCL+CLFFT
#------------------------------------------------------------------------------

if(GEARSHIFFT_BACKEND_CLFFT)
  # get library from vcpkg install
  if(GEARSHIFFT_USE_VCPKG)
    set(CLFFT_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
  endif()

  # can help to find opencl from nvidia
  if(GEARSHIFFT_BACKEND_CUFFT AND NOT ENV{CUDA_PATH})
    set(ENV{CUDA_PATH} ${CUDA_TOOLKIT_ROOT_DIR})
  endif()

  find_package(OpenCL ${_QUIET}) # cmake >= 3.7 provides imported targets
  if(OpenCL_FOUND)
    find_package(clFFT ${_QUIET})

    if(clFFT_FOUND)

      add_library(CLFFT INTERFACE)
      add_library(gearshifft::CLFFT ALIAS CLFFT)

      find_and_add_openmp(CLFFT ${_QUIET})

      target_compile_definitions(CLFFT INTERFACE CLFFT_ENABLED)
      target_link_libraries(CLFFT INTERFACE
        Common
        clFFT::clFFT
        OpenCL::OpenCL
        ${CMAKE_DL_LIBS}
        )

      message(STATUS " gearshifft::CLFFT enabled")
    endif()
  endif()
endif()
#------------------------------------------------------------------------------
# ROCM+ROCFFT
#------------------------------------------------------------------------------
if(GEARSHIFFT_BACKEND_ROCFFT)
  find_package(rocfft ${_QUIET}) # TODO: config via rocfft lib, e.g., /opt/rocm/rocfft/lib/cmake/rocfft
  if(rocfft_FOUND)

    find_package(hip ${_QUIET}) # needed for HC includes (target hip::hip_hcc does that)

    if(hip_FOUND AND hcc_DIR)
      if(NOT CMAKE_CXX_COMPILER MATCHES "hipcc$|hcc$")
        message(FATAL_ERROR "HCC required. Please rerun cmake with cmake -DCMAKE_CXX_COMPILER=hcc ... or disable ROCFFT backend.")
      endif()
      add_library(ROCFFT INTERFACE)
      add_library(gearshifft::ROCFFT ALIAS ROCFFT)

      # TODO: what is about that flag?
      include(CheckCCompilerFlag)
      include(CheckCXXCompilerFlag)
      check_cxx_compiler_flag(-Wno-unused-command-line-argument HAS_NO_WIGNORE_UNUSED_CLI_FLAG)
      if(HAS_NO_WIGNORE_UNUSED_CLI_FLAG)
        target_compile_options( ROCFFT INTERFACE -Wno-unused-command-line-argument )
      endif()

      # find_and_add_openmp(ROCFFT ${_QUIET})
      target_compile_definitions(ROCFFT INTERFACE ROCFFT_ENABLED)
      target_link_libraries(ROCFFT INTERFACE
        Common
        roc::rocfft
        hip::hip_hcc
        # ${CMAKE_DL_LIBS} # TODO: needed?
        )
      message(STATUS " gearshifft::ROCFFT enabled.")
    endif()
  endif()
endif()

#------------------------------------------------------------------------------
# FFTW
#------------------------------------------------------------------------------

if(GEARSHIFFT_BACKEND_FFTW)
  # get library from vcpkg install
  if(GEARSHIFFT_USE_VCPKG)
    if(NOT FFTW_USE_STATIC_LIBS)
      message(WARNING "Most linux vcpkg triplets use static linkage yet FFTW_USE_STATIC_LIBS is not ON.
                       gearshifft::FFTW may not enable.")
    endif()
    set(FFTW_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
  endif()

  if(GEARSHIFFT_USE_STATIC_LIBS)
    set(FFTW_USE_STATIC_LIBS ON) # otherwise find libs automatically
  endif()
  find_package(FFTW ${_QUIET})

  if(FFTW_FOUND)

    add_library(FFTW INTERFACE)
    add_library(gearshifft::FFTW ALIAS FFTW)

    set(GEARSHIFFT_BACKEND_FFTW_THREADS 0)

    if(GEARSHIFFT_BACKEND_FFTW_OPENMP)

      if(FFTW_OPENMP_LIBS)

        find_and_add_openmp(FFTW ${_QUIET})

        if(TARGET OpenMP::OpenMP_CXX)
          set(GEARSHIFFT_BACKEND_FFTW_THREADS 1)
        else()
          message(WARNING "Could not find OpenMP")
          set(GEARSHIFFT_BACKEND_FFTW_OPENMP OFF)
        endif()
      else()

        message(WARNING "FFTW libraries for OpenMP not found")
        set(GEARSHIFFT_BACKEND_FFTW_OPENMP OFF)
      endif() # FFTW_OPENMP_LIBS
    endif() # GEARSHIFFT_BACKEND_FFTW_OPENMP

    if(GEARSHIFFT_BACKEND_FFTW_PTHREADS)
      if(FFTW_THREADS_LIBS)
        find_package(Threads)
        set(GEARSHIFFT_BACKEND_FFTW_THREADS 1)
      else()
        message(WARNING "FFTW libraries for pthreads not found")
        set(GEARSHIFFT_BACKEND_FFTW_PTHREADS OFF)
      endif()
    endif()

    if(GEARSHIFFT_BACKEND_FFTW_PTHREADS AND GEARSHIFFT_BACKEND_FFTW_OPENMP)
      message(FATAL_ERROR "FFTW: Cannot use both OpenMP and pthreads, please disable one")
    endif()

    if(GEARSHIFFT_BACKEND_FFTW_PTHREADS
        OR GEARSHIFFT_BACKEND_FFTW_OPENMP
        OR FFTW_SERIAL_LIBS)

      target_include_directories(FFTW INTERFACE ${FFTW_INCLUDE_DIR})
      target_compile_definitions(FFTW INTERFACE
        FFTW_ENABLED
        GEARSHIFFT_BACKEND_FFTW_THREADS=${GEARSHIFFT_BACKEND_FFTW_THREADS}
        GEARSHIFFT_BACKEND_FFTW_OPENMP=${GEARSHIFFT_BACKEND_FFTW_OPENMP})
      target_link_libraries(FFTW INTERFACE
        Common
        $<$<STREQUAL:${GEARSHIFFT_BACKEND_FFTW_OPENMP},ON>:${FFTW_OPENMP_LIBS}>
        $<$<STREQUAL:${GEARSHIFFT_BACKEND_FFTW_PTHREADS},ON>:${FFTW_THREADS_LIBS}>
        ${FFTW_SERIAL_LIBS}
        $<$<BOOL:${Threads_FOUND}>:Threads::Threads>
        )
      message(STATUS " gearshifft::FFTW enabled.")
    endif()
  endif()
endif()

#------------------------------------------------------------------------------
# FFTW Wrappers of MKL
#------------------------------------------------------------------------------

if(GEARSHIFFT_BACKEND_FFTWWRAPPERS)
  find_package(FFTWWrappers ${_QUIET})
  if(FFTWWrappers_FOUND) # found FFTWWrappers for GNU or Intel

    if(FFTWWrappers_GNU_LIBRARIES AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU"
        AND FFTWWrappers_INTEL_LIBRARIES AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Intel")
      message(WARNING "Compiler does not match found FFTWWrappers")
    else()

      add_library(FFTWWrappers INTERFACE)
      add_library(gearshifft::FFTWWrappers ALIAS FFTWWrappers)

      # if compiler supports OpenMP then add flag, as FFTWWrappers might need it
      find_and_add_openmp(FFTWWrappers ${_QUIET})

      if(TARGET OpenMP::OpenMP_CXX)
        set(GEARSHIFFT_BACKEND_FFTW_THREADS 1)
      endif()

      target_include_directories(FFTWWrappers INTERFACE ${FFTWWrappers_MKL_INCLUDE_DIR})
      target_compile_definitions(FFTWWrappers INTERFACE
        FFTW_ENABLED
        GEARSHIFFT_BACKEND_FFTW_THREADS=${GEARSHIFFT_BACKEND_FFTW_THREADS})
      target_link_libraries(FFTWWrappers INTERFACE
        Common
        #      $<$<BOOL:${Threads_FOUND}>:Threads::Threads>
        )

      if(FFTWWrappers_GNU_LIBRARIES)
        target_link_libraries(FFTWWrappers INTERFACE
          $<$<CXX_COMPILER_ID:GNU>:${FFTWWrappers_GNU_LIBRARIES}>
          $<$<CXX_COMPILER_ID:Clang>:${FFTWWrappers_GNU_LIBRARIES}>
          )
      endif()

      if(FFTWWrappers_INTEL_LIBRARIES)
        target_link_libraries(FFTWWrappers INTERFACE
          $<$<CXX_COMPILER_ID:Intel>:${FFTWWrappers_INTEL_LIBRARIES}>
          )
      endif()

      target_link_libraries(FFTWWrappers INTERFACE
        $<$<CXX_COMPILER_ID:GNU>:-Wl,--start-group>
        ${FFTWWrappers_MKL_LIBRARIES}
        $<$<CXX_COMPILER_ID:GNU>:-Wl,--end-group>
        )
      message(STATUS " gearshifft::FFTWWrappers enabled.")

    endif() # compiler match fftwwrappers

  endif()
endif()

#------------------------------------------------------------------------------
# FFTW Wrappers for ESSL
#------------------------------------------------------------------------------

if(GEARSHIFFT_BACKEND_ESSL)

  find_package(ESSL)

  if(ESSL_FOUND)
    set(GEARSHIFFT_BACKEND_FFTW_THREADS 1)

    add_library(ESSL INTERFACE)
    add_library(gearshifft::ESSL ALIAS ESSL)

    target_compile_definitions(ESSL INTERFACE
      FFTW_ENABLED
      USE_ESSL
      GEARSHIFFT_BACKEND_FFTW_THREADS=${GEARSHIFFT_BACKEND_FFTW_THREADS}
    )

    target_link_libraries(ESSL INTERFACE
      ESSL::FFTW
      Common
    )

    message(STATUS " gearshifft::ESSL enabled.")
  endif()

endif()

#------------------------------------------------------------------------------
# FFTW Wrappers for ARM Performance Library
#------------------------------------------------------------------------------

if(GEARSHIFFT_BACKEND_ARMPL)

  find_package(ARMPL)

  if(ARMPL_FOUND)
    set(GEARSHIFFT_BACKEND_FFTW_THREADS 1)

    add_library(ARMPL INTERFACE)
    add_library(gearshifft::ARMPL ALIAS ARMPL)

    target_compile_definitions(ARMPL INTERFACE
      FFTW_ENABLED
      USE_ARMPL
      GEARSHIFFT_BACKEND_FFTW_THREADS=${GEARSHIFFT_BACKEND_FFTW_THREADS}
    )

    target_link_libraries(ARMPL INTERFACE
      ARMPL::ARMPL
      Common
    )

    message(STATUS " gearshifft::ARMPL enabled.")
  endif()

endif()

#------------------------------------------------------------------------------
# Eigen FFT
#------------------------------------------------------------------------------

if(GEARSHIFFT_BACKEND_EIGEN)
  if(GEARSHIFFT_BACKEND_EIGEN_USE_TEST)
    include("${GEARSHIFFT_ROOT}/cmake/modules/FindEigen3Mine.cmake")
  else()
    include("${GEARSHIFFT_ROOT}/cmake/modules/FindEigen3.cmake")
  endif()
  
  if(Eigen3_FOUND)
    add_library(EIGEN INTERFACE)
    
    target_link_libraries(EIGEN INTERFACE
      Common
      Eigen3::Eigen
    )
    target_compile_definitions(EIGEN INTERFACE
      EIGEN_ENABLED
    )
    
    # Eigen with KissFFT back-end
    add_library(EIGEN_KISSFFT INTERFACE)
    add_library(gearshifft::EIGEN_KISSFFT ALIAS EIGEN_KISSFFT)
    target_link_libraries(EIGEN_KISSFFT INTERFACE
      EIGEN
    )
    message(STATUS " gearshifft::EIGEN_KISSFFT enabled.")
  
    # Eigen with FFTW back-end
    if(FFTW_FOUND)
      add_library(EIGEN_FFTW INTERFACE)
      add_library(gearshifft::EIGEN_FFTW ALIAS EIGEN_FFTW)
      target_link_libraries(EIGEN_FFTW INTERFACE
        EIGEN
        FFTW
      )
      target_compile_options(EIGEN_FFTW INTERFACE
        -UFFTW_ENABLED # Undefine FFTW_ENABLED, linking against FFTW defines this
      )
      target_compile_definitions(EIGEN_FFTW INTERFACE
        EIGEN_FFTW_DEFAULT
      )
      message(STATUS " gearshifft::EIGEN_FFTW enabled.")
    endif(FFTW_FOUND)

    # Eigen with MKL back-end
    # set(ENABLE_CDFT ON)             # options for MKL find_package
    # set(MKL_THREADING "intel_thread") # todo: make this smarter to support intel compiler as options. Should also update eigen.hpp msg's
    find_package(MKL CONFIG ${_QUIET})
    
    if(MKL_FOUND)
      add_library(EIGEN_MKL INTERFACE)
      add_library(gearshifft::EIGEN_MKL ALIAS EIGEN_MKL)

      target_compile_options(EIGEN_MKL INTERFACE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
      target_include_directories(EIGEN_MKL INTERFACE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
      target_link_libraries(EIGEN_MKL INTERFACE $<LINK_ONLY:MKL::MKL> EIGEN)

      target_compile_definitions(EIGEN_MKL INTERFACE
        EIGEN_MKL_DEFAULT
      )

      message(STATUS " gearshifft::EIGEN_MKL enabled.")
    endif(MKL_FOUND)

    # Eigen with PocketFFT back-end
    if(GEARSHIFFT_USE_VCPKG)
      set(POCKETFFT_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
    endif()
    find_package(PocketFFT ${_QUIET})

    if(PocketFFT_FOUND)
      add_library(EIGEN_POCKETFFT INTERFACE)
      add_library(gearshifft::EIGEN_POCKETFFT ALIAS EIGEN_POCKETFFT)
      target_link_libraries(EIGEN_POCKETFFT INTERFACE
        EIGEN
      )
      target_compile_definitions(EIGEN_POCKETFFT INTERFACE
        EIGEN_POCKETFFT_DEFAULT
      )
      target_include_directories(EIGEN_POCKETFFT INTERFACE
        "${PocketFFT_INCLUDE_DIR}"
      )
      message(STATUS " gearshifft::EIGEN_POCKETFFT enabled.")
    endif(PocketFFT_FOUND)
  endif(Eigen3_FOUND)
endif(GEARSHIFFT_BACKEND_EIGEN)

#-------------------------------------------------------------------------------

if(NOT TARGET gearshifft::CUFFT
    AND NOT TARGET gearshifft::CLFFT
    AND NOT TARGET gearshifft::FFTW
    AND NOT TARGET gearshifft::FFTWWrappers
    AND NOT TARGET gearshifft::ROCFFT
    AND NOT TARGET gearshifft::ESSL
    AND NOT TARGET gearshifft::ARMPL
    AND NOT TARGET gearshifft::EIGEN
    )
  message(WARNING "No FFT backend found.")
endif()

#-------------------------------------------------------------------------------

add_subdirectory(src gearshifft)

include(CTest) # defines: BUILD_TESTING, default is ON

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

#-------------------------------------------------------------------------------

# installs config extents files

install(
  DIRECTORY share/gearshifft
  DESTINATION share
  COMPONENT config
  FILES_MATCHING
  PATTERN *.conf
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ
              WORLD_READ
  )
install(
  FILES README.md
  DESTINATION doc
  COMPONENT doc
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ
              WORLD_READ
  )

#-------------------------------------------------------------------------------

# packaging
# check that we are in top-level CMakeLists
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_subdirectory(packaging)
  # include(CPack) will happen in here
endif()
