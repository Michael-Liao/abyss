# add_library(abyss-ops SHARED)
add_library(abyss-ops OBJECT)

set(ABYSS_OPS_HEADERS
  "dispatcher.h"
  # "visitor.h"
  # "operation.h"
  "dtype_ops.h"
  "util_ops.h"
  "merge_ops.h"
  "vector_ops.h"
  "matrix_ops.h"
  "conversion_ops.h"
  )

set(ABYSS_OPS_SOURCES
  "dtype_ops.cc"
  "util_ops.cc"
  "merge_ops.cc"
  "vector_ops.cc"
  "matrix_ops.cc"
  # "comp_ops.cc"
  )

target_sources(abyss-ops
  PRIVATE
    ${ABYSS_OPS_HEADERS}
    ${ABYSS_OPS_SOURCES}
  )
target_compile_features(abyss-ops PUBLIC cxx_std_14)

set_property(TARGET abyss-ops
  PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(abyss-ops
  PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include;${PROJECT_BINARY_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/abyss>"
  PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    # "${PROJECT_BINARY_DIR}"
  )

target_link_libraries(abyss-ops
  PRIVATE
    abyss-backend
    abyss-core
  )


# install(TARGETS abyss-ops
#   EXPORT AbyssTargets
#   LIBRARY
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     COMPONENT Abyss_Runtime
#     # NAMELINK_COMPONENT Abyss_Development
#   )