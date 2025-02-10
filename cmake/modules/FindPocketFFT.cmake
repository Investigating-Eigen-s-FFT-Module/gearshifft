# Searches for the PocketFFT header for gearshifft::EIGEN_POCKETFFT
# PocketFFT source: https://github.com/hayguen/pocketfft/tree/cpp
#
# This module defines:
#  PocketFFT_FOUND        - True if PocketFFT is found
#  PocketFFT_INCLUDE_DIR  - The PocketFFT include directory
#
# The following variables can be set as arguments:
#  POCKETFFT_DIR         - Root directory to search for PocketFFT (overrides default search)

# Default search paths if POCKETFFT_DIR is not set
if(POCKETFFT_DIR)
  find_path(POCKETFFT_ROOT_DIR
    NAMES "include/pocketfft_hdronly.h"
    PATHS "${POCKETFFT_DIR}"
    PATHS ENV POCKETFFT_DIR
    DOC "PocketFFT root directory."
    NO_DEFAULT_PATH)
else()
  find_path(POCKETFFT_ROOT_DIR
    NAMES "include/pocketfft_hdronly.h"
    DOC "PocketFFT root directory.")
endif()

# Find header
find_path(PocketFFT_INCLUDE_DIR
  NAMES "pocketfft_hdronly.h"
  PATHS "${POCKETFFT_ROOT_DIR}"
  PATH_SUFFIXES
    "include"
    "include/pocketfft"
  DOC "PocketFFT include directory"
  NO_DEFAULT_PATH
)

# Handle QUIET and REQUIRED arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PocketFFT
  REQUIRED_VARS PocketFFT_INCLUDE_DIR
  FAIL_MESSAGE "Could NOT find PocketFFT"
)

if(NOT PocketFFT_FIND_QUIETLY)
    message("++ FindPocketFFT")
    message("++ PocketFFT_INCLUDE_DIR : ${PocketFFT_INCLUDE_DIR}")
endif()

# Advanced variables shouldn't show up in GUI by default
mark_as_advanced(
  PocketFFT_INCLUDE_DIR
  POCKETFFT_ROOT_DIR
)
