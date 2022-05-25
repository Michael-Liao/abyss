find_package(Catch2 REQUIRED)

include(CTest)
include(Catch)

add_executable(abyss-test
  "test_main.cc"
  "test_array.cc"
  "test_utility.cc"
  "test_tensor.cc"
  "test_functional.cc"
  )

  add_subdirectory("backend/native")
  # add_subdirectory("ops")
  add_subdirectory("autograd")

# message("${PROJECT_SOURCE_DIR}")

target_include_directories(abyss-test
  PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_BINARY_DIR}"
  )

target_link_libraries(abyss-test
  PRIVATE
    Catch2::Catch2
    abyss-backend
    abyss-core
    abyss-ops
    abyss
  )
# set_target_properties(abyss-test
#   PROPERTIES
#     CXX_VISIBILITY_PRESET default
#     VISIBILITY_INLINES_HIDDEN OFF)

catch_discover_tests(abyss-test)
