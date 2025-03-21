# - Find the FFTW library
#
# Usage:
#   find_package(FFTWWrappers [REQUIRED] [QUIET])
#
# It sets the following variables:
#   FFTWWrappers_FOUND             ... true if fftw is found on the system
#   FFTWWrappers_GNU_LIBRARIES     ... list of library that were build with the GNU binary layout
#   FFTWWrappers_INTEL_LIBRARIES   ... list of library that were build with the intel binary layout
#   FFTWWrappers_MSVS_LIBRARIES    ... list of library that were build with the MSVS binary layout
#   FFTWWrappers_LIBRARIES         ... list of library identifiers that were found (will be filled with serial/openmp/pthreads enabled file names if present, only stubs will be filled in not full paths)
#   FFTWWrappers_MKL_LIBRARIES     ... list of library in the MKL that need to be linked to
#   FFTWWrappers_MKL_INCLUDE_DIR   ... folder containing fftw3.h
#   FFTWWrappers_MKL_LIBRARY_DIRS  ... folder containing the libraries in FFTWWrappers_MKL_LIBRARIES
#   FFTWWrappers_LIBRARY_DIR	   ... fftw library directory
#
# The following variables will be checked by the function
#   FFTWWrappers_DIR              ... if set, the libraries are exclusively searched
#                                      under this path
#   MKL_DIR                       ... take the MKL libraries from here
#

if(GEARSHIFFT_USE_STATIC_LIBS)
  set(PREFERENCE_LIBRARY_PREFIX "${CMAKE_STATIC_LIBRARY_PREFIX}")
  set(PREFERENCE_LIBRARY_SUFFIX "${CMAKE_STATIC_LIBRARY_SUFFIX}")
  message("FFTWWrappers will prefer libraries with prefix '${PREFERENCE_LIBRARY_PREFIX}' and suffix '${PREFERENCE_LIBRARY_SUFFIX}'")
else()
  set(PREFERENCE_LIBRARY_PREFIX)
  set(PREFERENCE_LIBRARY_SUFFIX)
  message("FFTWWrappers will prefer libraries with no prefix and no suffix")
endif()

# If environment variable FFTWWrappers_DIR is defined, it has the same effect as the cmake variable
if( NOT FFTWWrappers_DIR AND DEFINED ENV{FFTWWrappers_DIR} )
  if( EXISTS "$ENV{FFTWWrappers_DIR}/" )
    set( FFTWWrappers_DIR $ENV{FFTWWrappers_DIR} )
  else()
    message( "FFTWWrappers_DIR set to ${FFTWWrappers_DIR}, but folder does not exist")
  endif()
endif()

# If environment variable MKL_DIR is defined, it has the same effect as the cmake variable
if( NOT MKL_DIR AND DEFINED ENV{MKLROOT})
  if( EXISTS "$ENV{MKLROOT}/" )
    set( MKL_DIR $ENV{MKLROOT} )
  else()
    message( "MKLROOT set to $ENV{MKLROOT}, but folder does not exist")
  endif()
endif()

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(MKL_INTERFACE_LIBDIR "intel64")
    set(MKL_INTERFACE_LIBNAME "mkl_intel_lp64")
else()
    set(MKL_INTERFACE_LIBDIR "ia32")
    set(MKL_INTERFACE_LIBNAME "mkl_intel")
endif()

################################### FFTWWrappers related ##################################

#initialize library variables
find_library(
  FFTWWrappers_GNU_LIBRARIES
  NAMES ${PREFERENCE_LIBRARY_PREFIX}fftw3xc_gnu${PREFERENCE_LIBRARY_SUFFIX} fftw3xc_gnu
  PATHS ${FFTWWrappers_DIR}
  PATH_SUFFIXES "lib" "lib64"
  NO_DEFAULT_PATH
  )
find_library(
  FFTWWrappers_GNU_LIBRARIES
  NAMES ${PREFERENCE_LIBRARY_PREFIX}fftw3xc_gnu${PREFERENCE_LIBRARY_SUFFIX} fftw3xc_gnu
  PATHS ${MKL_DIR}
  PATH_SUFFIXES "lib/${MKL_INTERFACE_LIBDIR}_lin" "lib/${MKL_INTERFACE_LIBDIR}_mac" "lib/${MKL_INTERFACE_LIBDIR}"
  NO_DEFAULT_PATH
  )
find_library(
  FFTWWrappers_GNU_LIBRARIES
  NAMES ${PREFERENCE_LIBRARY_PREFIX}fftw3xc_gnu${PREFERENCE_LIBRARY_SUFFIX} fftw3xc_gnu
  )

if(EXISTS ${FFTWWrappers_GNU_LIBRARIES})
  get_filename_component(FFTWWrappers_LIBRARY_DIR ${FFTWWrappers_GNU_LIBRARIES} DIRECTORY)
endif()

find_library(
  FFTWWrappers_INTEL_LIBRARIES
  NAMES ${PREFERENCE_LIBRARY_PREFIX}fftw3xc_intel${PREFERENCE_LIBRARY_SUFFIX} fftw3xc_intel
  PATHS ${FFTWWrappers_DIR}
  PATH_SUFFIXES "lib" "lib64"
  NO_DEFAULT_PATH
  )
find_library(
  FFTWWrappers_INTEL_LIBRARIES
  NAMES ${PREFERENCE_LIBRARY_PREFIX}fftw3xc_intel${PREFERENCE_LIBRARY_SUFFIX} fftw3xc_intel
  PATHS ${MKL_DIR}
  PATH_SUFFIXES "lib/${MKL_INTERFACE_LIBDIR}_lin" "lib/${MKL_INTERFACE_LIBDIR}_mac" "lib/${MKL_INTERFACE_LIBDIR}_win" "lib/${MKL_INTERFACE_LIBDIR}"
  NO_DEFAULT_PATH
)
find_library(
  FFTWWrappers_INTEL_LIBRARIES
  NAMES ${PREFERENCE_LIBRARY_PREFIX}fftw3xc_intel${PREFERENCE_LIBRARY_SUFFIX} fftw3xc_intel
)

if(EXISTS ${FFTWWrappers_INTEL_LIBRARIES})
  get_filename_component(FFTWWrappers_LIBRARY_DIR ${FFTWWrappers_INTEL_LIBRARIES} DIRECTORY)
endif()

find_library(
  FFTWWrappers_MSVS_LIBRARIES
  NAMES fftw3xc_msvs
  PATHS ${FFTWWrappers_DIR}
  PATH_SUFFIXES "lib" "lib64"
  NO_DEFAULT_PATH
  )
find_library(
  FFTWWrappers_MSVS_LIBRARIES
  NAMES fftw3xc_msvs
  PATHS ${MKL_DIR}
  PATH_SUFFIXES "lib/${MKL_INTERFACE_LIBDIR}_win" "lib/${MKL_INTERFACE_LIBDIR}"
  NO_DEFAULT_PATH
  )
find_library(
  FFTWWrappers_MSVS_LIBRARIES
  NAMES fftw3xc_msvs
  )

if(EXISTS ${FFTWWrappers_MSVS_LIBRARIES})
  get_filename_component(FFTWWrappers_LIBRARY_DIR ${FFTWWrappers_MSVS_LIBRARIES} DIRECTORY)
endif()

######################################### MKL related #####################################

find_file(
  FFTWWrapper_include_file
  NAMES fftw3.h
  PATHS ${MKL_DIR} ${MKL_DIR}/include/fftw
  PATH_SUFFIXES "include" "include/fftw"
  NO_DEFAULT_PATH
  )

if(EXISTS ${FFTWWrapper_include_file})
  get_filename_component(FFTWWrappers_MKL_INCLUDE_DIR ${FFTWWrapper_include_file} DIRECTORY)
else()
  message( "FFTWWrappers was not able to find fftw3.h in ${MKL_DIR}")
endif()

find_library(
  MKL_INTEL
  NAMES ${PREFERENCE_LIBRARY_PREFIX}${MKL_INTERFACE_LIBNAME}${PREFERENCE_LIBRARY_SUFFIX} ${MKL_INTERFACE_LIBNAME}
  PATHS ${MKL_DIR}
  PATH_SUFFIXES "lib" "lib/${MKL_INTERFACE_LIBDIR}_lin" "lib/${MKL_INTERFACE_LIBDIR}_mac" "lib/${MKL_INTERFACE_LIBDIR}_win" "lib/${MKL_INTERFACE_LIBDIR}"
  NO_DEFAULT_PATH
  )

if(EXISTS ${MKL_INTEL})
  list(APPEND FFTWWrappers_MKL_LIBRARIES "${MKL_INTEL}")
else()
  message( "FFTWWrappers was not able to find ${MKL_INTERFACE_LIBNAME} in ${MKL_DIR}")
endif()

find_library(
  MKL_INTEL_THREAD
  NAMES ${PREFERENCE_LIBRARY_PREFIX}mkl_intel_thread${PREFERENCE_LIBRARY_SUFFIX} mkl_intel_thread
  PATHS ${MKL_DIR}
  PATH_SUFFIXES "lib" "lib/${MKL_INTERFACE_LIBDIR}_lin" "lib/${MKL_INTERFACE_LIBDIR}_mac" "lib/${MKL_INTERFACE_LIBDIR}_win" "lib/${MKL_INTERFACE_LIBDIR}"
  NO_DEFAULT_PATH
  )

if(EXISTS ${MKL_INTEL_THREAD})
  list(APPEND FFTWWrappers_MKL_LIBRARIES "${MKL_INTEL_THREAD}")
else()
  message( "FFTWWrappers was not able to find mkl_intel_thread in ${MKL_DIR}")
endif()

find_library(
  MKL_CORE
  NAMES ${PREFERENCE_LIBRARY_PREFIX}mkl_core${PREFERENCE_LIBRARY_SUFFIX} mkl_core
  PATHS ${MKL_DIR}
  PATH_SUFFIXES "lib" "lib/${MKL_INTERFACE_LIBDIR}_lin" "lib/${MKL_INTERFACE_LIBDIR}_mac" "lib/${MKL_INTERFACE_LIBDIR}_win" "lib/${MKL_INTERFACE_LIBDIR}"
  NO_DEFAULT_PATH
  )

if(EXISTS ${MKL_CORE})
  list(APPEND FFTWWrappers_MKL_LIBRARIES "${MKL_CORE}")
else()
  message("FFTWWrappers was not able to find mkl_core in ${MKL_DIR}")
endif()

list(APPEND FFTWWrappers_MKL_LIBRARY_DIRS "${MKL_DIR}/../compiler/lib/${MKL_INTERFACE_LIBDIR}")
list(APPEND FFTWWrappers_MKL_LIBRARY_DIRS "${MKL_DIR}/../tbb/lib/${MKL_INTERFACE_LIBDIR}/gcc4.4")

# NOTE: According to Intel documentation, it is generally not recommended to
# link against the static variants of the openMP libraries so we put them last:
find_library(
  MKL_IOMP5
  NAMES iomp5 libiomp5.a libiomp5md
  PATHS ${MKL_DIR} ${MKL_DIR}/../compiler ${MKL_DIR}/../tbb
  PATH_SUFFIXES "lib" "lib/${MKL_INTERFACE_LIBDIR}_lin" "lib/${MKL_INTERFACE_LIBDIR}_mac" "lib/${MKL_INTERFACE_LIBDIR}_win" "lib/${MKL_INTERFACE_LIBDIR}" "lib/${MKL_INTERFACE_LIBDIR}/gcc4.4"
  NO_DEFAULT_PATH
  )

if(EXISTS ${MKL_IOMP5})
  list(APPEND FFTWWrappers_MKL_LIBRARIES "${MKL_IOMP5}")
else()

endif()

list(APPEND FFTWWrappers_MKL_LIBRARIES
     $<$<NOT:$<C_COMPILER_ID:MSVC>>:m> "${CMAKE_DL_LIBS}")

if(NOT FFTWWrappers_FIND_QUIETLY)
  message("++ FindFFTWWrappers")
  message("++ FFTWWrappers_GNU_LIBRARIES    : ${FFTWWrappers_GNU_LIBRARIES}")
  message("++ FFTWWrappers_INTEL_LIBRARIES  : ${FFTWWrappers_INTEL_LIBRARIES}")
  message("++ FFTWWrappers_MSVS_LIBRARIES   : ${FFTWWrappers_MSVS_LIBRARIES}")
  message("++ FFTWWrappers_LIBRARY_DIR      : ${FFTWWrappers_LIBRARY_DIR}")
  message("++ FFTWWrappers_MKL_LIBRARIES    : ${FFTWWrappers_MKL_LIBRARIES}")
  message("++ FFTWWrappers_MKL_LIBRARY_DIRS : ${FFTWWrappers_MKL_LIBRARY_DIRS}")
  message("++ FFTWWrappers_MKL_INCLUDE_DIR  : ${FFTWWrappers_MKL_INCLUDE_DIR}")
endif()


######################################### EXPORTS #####################################

include(FindPackageHandleStandardArgs)

if(FFTWWrappers_GNU_LIBRARIES)
  find_package_handle_standard_args(FFTWWrappers
    REQUIRED_VARS FFTWWrappers_GNU_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_INCLUDE_DIR
    )
  set(FFTWWrappers_LIBRARIES "${FFTWWrappers_GNU_LIBRARIES}")
elseif(FFTWWrappers_INTEL_LIBRARIES)
  find_package_handle_standard_args(FFTWWrappers
    REQUIRED_VARS FFTWWrappers_INTEL_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_INCLUDE_DIR
    )
  set(FFTWWrappers_LIBRARIES "${FFTWWrappers_INTEL_LIBRARIES}")
elseif(FFTWWrappers_MSVS_LIBRARIES)
  find_package_handle_standard_args(FFTWWrappers
    REQUIRED_VARS FFTWWrappers_MSVS_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_LIBRARIES
    REQUIRED_VARS FFTWWrappers_MKL_INCLUDE_DIR
    )
  set(FFTWWrappers_LIBRARIES "${FFTWWrappers_MSVS_LIBRARIES}")
endif()


mark_as_advanced(
  FFTWWrappers_LIBRARIES
  FFTWWrappers_GNU_LIBRARIES
  FFTWWrappers_INTEL_LIBRARIES
  FFTWWrappers_MSVS_LIBRARIES
  FFTWWrappers_LIBRARY_DIR
  FFTWWrappers_MKL_LIBRARIES
  FFTWWrappers_MKL_LIBRARY_DIR
  FFTWWrappers_MKL_INCLUDE_DIR
)
