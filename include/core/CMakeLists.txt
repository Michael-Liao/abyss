add_library(abyss-core INTERFACE)

set(ABYSS_CORE_HEADERS
  "allocator.h"
  "iterator.h"
  "dtype.h"
  "traits.h"
  "array.h"
  "visitor.h"
  "utility.h"
  )

target_sources(abyss-core
  INTERFACE
    ${ABYSS_CORE_HEADERS}
  )

target_include_directories(abyss-core
  INTERFACE
    "${PROJECT_BINARY_DIR}"
    # "${PROJECT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )

install(FILES
  ${ABYSS_CORE_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/abyss/core
  )