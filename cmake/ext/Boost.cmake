#---------------------------------------------------------------------------
# Get and build boost, if it is not already present

set(GEARSHIFFT_EXT_BOOST_VERSION "1.68.0" CACHE STRING "Boost version to be built.")
set_property(CACHE GEARSHIFFT_EXT_BOOST_VERSION PROPERTY STRINGS "1.65.1;1.66.0;1.68.0")
set(BOOST_ROOT ${GEARSHIFFT_SUPERBUILD_EXT_DIR}/boost/ CACHE PATH "'Boost' root directory")

find_path(Boost_INCLUDE_DIR
  NAMES boost/config.hpp
  PATHS ${BOOST_ROOT}/include
  NO_DEFAULT_PATH)

find_library(Boost_LIBRARY_DIR
  NAMES boost_program_options
  PATHS ${BOOST_ROOT}/lib
  NO_DEFAULT_PATH)

if((NOT Boost_INCLUDE_DIR) OR (NOT EXISTS ${Boost_INCLUDE_DIR})
    OR
    (NOT Boost_LIBRARY_DIR) OR (NOT EXISTS ${Boost_LIBRARY_DIR}))

  message("'Boost' library could not be found, so [make] will download and build it.")

  set( Boost_Bootstrap_Command )
  string(REPLACE "." "_" GEARSHIFFT_EXT_BOOST_VERSION_U ${GEARSHIFFT_EXT_BOOST_VERSION})

  # md5 hashes of tar.gz files
  if( UNIX OR APPLE )
    if(GEARSHIFFT_EXT_BOOST_VERSION MATCHES "1.68.0")
      set(Boost_hash "5d8b4503582fffa9eefdb9045359c239")
    elseif(GEARSHIFFT_EXT_BOOST_VERSION MATCHES "1.66.0")
      set(Boost_hash "d275cd85b00022313c171f602db59fc5")
    elseif(GEARSHIFFT_EXT_BOOST_VERSION MATCHES "1.65.1")
      set(Boost_hash "ee64fd29a3fe42232c6ac3c419e523cf")
    else()
      set(Boost_hash "")
    endif()

    set( Boost_url "http://sourceforge.net/projects/boost/files/boost/${GEARSHIFFT_EXT_BOOST_VERSION}/boost_${GEARSHIFFT_EXT_BOOST_VERSION_U}.tar.gz")
    set( Boost_Bootstrap_Command ./bootstrap.sh )
    set( Boost_b2_Command ./b2 )
  elseif(WIN32)
    #   set( Boost_url "http://sourceforge.net/projects/boost/files/boost/1.59.0/boost_1_59_0.zip")
    #   set( Boost_sha1 ${GEARSHIFFT_EXT_BOOST_SHA1} )
    #   set( Boost_Bootstrap_Command cmd /C bootstrap.bat msvc )
    #   set( Boost_b2_Command b2.exe )
    # endif()
  endif()

  if(GEARSHIFFT_USE_STATIC_LIBS)
    set(BUILD_LIBS "link=static" "define=BOOST_TEST_NO_MAIN" "define=BOOST_TEST_ALTERNATIVE_INIT_API")
  else()
    set(BUILD_LIBS "link=shared")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(BOOST_TOOLSET "--with-toolset=clang")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(BOOST_TOOLSET "--with-toolset=intel-linux")
  endif()
  # currently builds always in release
  ExternalProject_Add(Boost
    BUILD_IN_SOURCE 1
    URL ${Boost_url}
    URL_MD5 ${Boost_hash}
    PREFIX ${GEARSHIFFT_SUPERBUILD_EXT_DIR}
    SOURCE_DIR ${GEARSHIFFT_SUPERBUILD_EXT_SOURCE_DIR}/boost
    DOWNLOAD_DIR ${GEARSHIFFT_ROOT}/ext/downloads/
    UPDATE_COMMAND ""
    # --with-toolset=
    CONFIGURE_COMMAND CC=${CMAKE_C_COMPILER}; CXX=${CMAKE_CXX_COMPILER} ${Boost_Bootstrap_Command} ${BOOST_TOOLSET} --with-libraries=program_options,filesystem,system,test
    # -d0 Suppress all informational messages
    BUILD_COMMAND ${Boost_b2_Command} -j8 -d0 --prefix=${BOOST_ROOT} address-model=64 ${BUILD_LIBS} variant=release install
    INSTALL_COMMAND ""
    )
  # defines in b2 command above are required, otherwise linking will fail (boost::unit_test::unit_test_main)
  # https://www.boost.org/doc/libs/1_68_0/libs/test/doc/html/boost_test/adv_scenarios/static_lib_customizations/entry_point.html

  list(APPEND gearshifft_DEPENDENCIES Boost)

endif()
