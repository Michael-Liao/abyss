add_library(abyss-core INTERFACE)

target_sources(abyss-core
  INTERFACE
    "allocator.h"
    "iterator.h"
    "dtype.h"
    "traits.h"
    "array.h"
    # "visitable.h"
    "visitor.h"
    # "expression.h"
  )

target_include_directories(abyss-core
  INTERFACE
    "${PROJECT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
