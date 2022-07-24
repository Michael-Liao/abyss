add_library(abyss-backend SHARED)

find_package(BLAS REQUIRED)

set(ABYSS_BACKEND_HEADERS
  "arithmetics.h"
  "matmul.h"
  "comparison.h"
  "amath.h"
  "reduction.h"
  )
# todo: use different target sources for different compilers so we can swap backends
set(ABYSS_BACKEND_SOURCES
  "native/arithmetics.cc"
  "native/matmul.cc"
  "native/comparison.cc"
  "native/amath.cc"
  "native/reduction.cc"
  )

# message("${ABYSS_PUBLIC_HEADERS}")

target_sources(abyss-backend
  PRIVATE
    ${ABYSS_BACKEND_HEADERS}
    ${ABYSS_BACKEND_SOURCES}
  )
# set_property(TARGET abyss-backend
#   PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(abyss-backend PUBLIC cxx_std_14)
target_include_directories(abyss-backend
  PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/abyss>"
  PRIVATE
    "${PROJECT_SOURCE_DIR}/thrid_party"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    # "${PROJECT_BINARY_DIR}"
  )

# imported target BLAS::BLAS is introducted in 3.18
target_link_libraries(abyss-backend
  PRIVATE
    ${BLAS_LIBRARIES}
)

# install(TARGETS abyss-backend
#   EXPORT AbyssTargets
#   LIBRARY
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     COMPONENT Abyss_Runtime
#     # NAMELINK_COMPONENT Abyss_Development
#   )