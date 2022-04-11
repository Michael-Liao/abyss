cmake_minimum_required(VERSION 3.15)

project(abyss
  VERSION 0.1.0
  DESCRIPTION "a multi-deimensional array library"
  LANGUAGES CXX
  )

include(GNUInstallDirs)
include(CheckLanguage)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  # cuda compile features are introduced for 3.17
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
  message("no CUDA compiler found. Skipping GPU support")
endif()

# find submodule-like dependencies in tree (probably not useful for now)
if(NOT APPLE)
  # this has to be defined before all targets
  set(CMAKE_INSTALL_RPATH $ORIGIN)
endif()

# header symbols should be hidden by default
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

### Library Definition ###
add_subdirectory("src/backend")
add_subdirectory("include/core")
add_subdirectory("src/ops")

set(ABYSS_PUBLIC_HEADERS  
  "include/types.h"
  "include/tensor.h"
  "include/functional.h"
  "include/operators.h"
  )

add_library(abyss SHARED
  # ${ABYSS_PUBLIC_HEADERS}
  "src/types.cc"
  "src/tensor.cc"
  "src/functional.cc"
  "src/operators.cc"
  )

target_sources(abyss
  PRIVATE
    ${ABYSS_PUBLIC_HEADERS}
  )

target_compile_features(abyss PUBLIC cxx_std_14)
set_target_properties(abyss
  PROPERTIES
    CXX_EXTENSION OFF
    SOVERSION ${abyss_VERSION_MAJOR}
    VERSION ${abyss_VERSION}
  )
target_include_directories(abyss
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include;${PROJECT_BINARY_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/abyss>"
  PRIVATE
    "${PROJECT_SOURCE_DIR}/thrid_party"
    "${PROJECT_SOURCE_DIR}/src"
    # "${PROJECT_BINARY_DIR}"
  )

target_link_libraries(abyss
  PRIVATE
    abyss-core
    abyss-ops
  )

### Testing ###
option(BUILD_TESTS "build tests (requires Catch2)" ON)
if(BUILD_TESTS)
  # message("debug build")
  add_subdirectory("tests")
endif()

### Generating documentation ###
option(BUILD_DOCS "build documentation (requires Doxygen)" OFF)
if (BUILD_DOCS)
  find_package(Doxygen 1.8 REQUIRED)
  doxygen_add_docs(abyss-docs ALL)
endif()


# generate ABYSS_EXPORT macro to expose symbols that are part of the API
include(GenerateExportHeader)
generate_export_header(abyss)

### Installation Strategies ###

# configure the library and install
install(TARGETS abyss abyss-backend
  EXPORT AbyssTargets
  RUNTIME
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Abyss_Runtime
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT Abyss_Runtime
    NAMELINK_COMPONENT Abyss_Development
  ARCHIVE
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT Abyss_Development
  INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/abyss
  )
# install header files
install(FILES
  ${ABYSS_PUBLIC_HEADERS}
  "${PROJECT_BINARY_DIR}/abyss_export.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/abyss
  )
# generate the export file
install(EXPORT AbyssTargets
  FILE AbyssTargets.cmake
  NAMESPACE Abyss::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/abyss
  )
# alias is created for people who are using my project as a submodule
add_library(Abyss::abyss ALIAS abyss)

include(CMakePackageConfigHelpers)
# generate the `ProjConfig.cmake` file
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/AbyssConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/abyss
  )
# generate the `ProjConfigVersion.cmake` file for version match
write_basic_package_version_file(
  AbyssConfigVersion.cmake
  VERSION ${abyss_VERSION}
  COMPATIBILITY SameMajorVersion
  )

# install the cmake files so other people can use `find_package` to find it
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/AbyssConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/AbyssConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/abyss
  )
