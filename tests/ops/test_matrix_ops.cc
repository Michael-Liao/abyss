#include <tuple>
#include <vector>

#include "catch2/catch.hpp"
#include "core/array.h"
#include "core/utility.h"
#include "ops/matrix_ops.h"

// shape1: 4 x 9 x 3 x 2
// shape2:         1 x 2 x 5
// output: 4 x 9 x 3 x 5

TEST_CASE("Matmul Visitor output and broadcast scheme",
          "[matmul][output_shape][broadcast]") {
  using namespace abyss::core;
  std::vector<int> shape1 = {4, 9, 3, 2};
  std::vector<int> shape2 = {2, 5};

  std::vector<int> output_shape, broadcast_shape1, broadcast_shape2;

  std::vector<int> tgt_output_shape = {4, 9, 3, 5};
  std::vector<int> tgt_broadcast_shape1 = {4, 9, 3, 2};
  std::vector<int> tgt_broadcast_shape2 = {4, 9, 2, 5};
  SECTION("basic matmul broadcast") {
    REQUIRE_NOTHROW(std::tie(output_shape, broadcast_shape1, broadcast_shape2) =
                        MatmulVisitor::calc_output_shape(shape1, shape2));

    REQUIRE(output_shape == tgt_output_shape);
    REQUIRE(broadcast_shape1 == tgt_broadcast_shape1);
    REQUIRE(broadcast_shape2 == tgt_broadcast_shape2);
  }

  SECTION("with leading dimensions") {
    shape2 = {1, 2, 5};

    REQUIRE_NOTHROW(std::tie(output_shape, broadcast_shape1, broadcast_shape2) =
                        MatmulVisitor::calc_output_shape(shape1, shape2));

    REQUIRE(output_shape == tgt_output_shape);
    REQUIRE(broadcast_shape1 == tgt_broadcast_shape1);
    REQUIRE(broadcast_shape2 == tgt_broadcast_shape2);
  }

  SECTION("throws exception if not broadcastable") {
    shape2 = {2, 2, 5};
    REQUIRE_THROWS(MatmulVisitor::calc_output_shape(shape1, shape2));
  }

  SECTION("throws exception if common shape doesn't match") {
    shape2 = {1, 3, 5};
    REQUIRE_THROWS(MatmulVisitor::calc_output_shape(shape1, shape2));
  }
}

// TEST_CASE("matrix multiplication", "[matmul][visit]") {
//   using namespace abyss::core;

//   // std::vector<int> shape1 = {3, 3, 2};
//   TensorDesc desc1{0, {3, 3, 2}, shape2strides({3, 3, 2})};
//   ArrayImpl<int32_t> arr1 = {1, 2, 3, 4, 5, 6, 1, 2, 3,
//                              4, 5, 6, 1, 2, 3, 4, 5, 6};

//   ArrayImpl<int32_t>* result;

//   SECTION("basic case without broadcast") {
//     // std::vector<int> shape2 = {3, 2, 3};
//     TensorDesc desc2{0, {3, 2, 3}, shape2strides({3, 2, 3})};
//     ArrayImpl<int32_t> arr2 = {1, 1, 1, 1, 1, 1, 2, 2, 2,
//                                2, 2, 2, 3, 3, 3, 3, 3, 3};

//     ArrayImpl<int32_t> target = {3, 3, 3, 7,  7,  7,  11, 11, 11,
//                                  6, 6, 6, 14, 14, 14, 22, 22, 22,
//                                  9, 9, 9, 21, 21, 21, 33, 33, 33};

//     MatmulVisitor vis(desc1, desc2);
//     vis.visit(&arr1, &arr2);

//     result = dynamic_cast<ArrayImpl<int32_t>*>(vis.data());

//     REQUIRE(result->size() == 27);
//     for (size_t i = 0; i < result->size(); i++) {
//       INFO("index: " << i);
//       CHECK(result->at(i) == target.at(i));
//     }
//   }

//   SECTION("matmul with broadcasting") {
//     // std::vector<int> shape2 = {2, 3};
//   TensorDesc desc2{0, {2, 3}, shape2strides({2, 3})};
//     ArrayImpl<int32_t> arr2 = {1, 1, 1, 1, 1, 1};

//     ArrayImpl<int32_t> target = {3, 3, 3, 7, 7, 7, 11, 11, 11,
//                                  3, 3, 3, 7, 7, 7, 11, 11, 11,
//                                  3, 3, 3, 7, 7, 7, 11, 11, 11};

//     MatmulVisitor vis(desc1, desc2);
//     vis.visit(&arr1, &arr2);

//     result = dynamic_cast<ArrayImpl<int32_t>*>(vis.data());

//     REQUIRE(result->size() == 27);
//     for (size_t i = 0; i < result->size(); i++) {
//       INFO("index: " << i);
//       CHECK(result->at(i) == target.at(i));
//     }
//   }
// }