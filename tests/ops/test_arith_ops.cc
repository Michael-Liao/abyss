#include <catch2/catch.hpp>

#include "core/utility.h"
#include "ops/vector_ops.h"

#include "core/dispatcher.h"
#include "ops/dtype_ops.h"
#include "scalartype.h"
#include "tensor.h"

TEST_CASE("test broadcast shape calculations", "[broadcast][arith]") {
  using namespace abyss::core;

  std::vector<int> shape2 = {3, 3, 2};
  std::vector<int> output_shape;

  SECTION("basic broadcast shape") {
    std::vector<int> shape1 = {3, 1, 2};
    output_shape = AddVisitor::calc_output_shape(shape1, shape2);

    REQUIRE(output_shape == shape2);
  }

  SECTION("broadcast with pending 1") {
    std::vector<int> shape1 = {1, 2};

    output_shape = AddVisitor::calc_output_shape(shape1, shape2);
    REQUIRE(output_shape == shape2);
    output_shape = AddVisitor::calc_output_shape(shape2, shape1);
    REQUIRE(output_shape == shape2);
  }

  SECTION("column vector") {
    std::vector<int> shape1 = {3, 1};

    output_shape = AddVisitor::calc_output_shape(shape1, shape2);
    REQUIRE(output_shape == shape2);
    output_shape = AddVisitor::calc_output_shape(shape2, shape1);
    REQUIRE(output_shape == shape2);
  }

  SECTION("row vector") {
    std::vector<int> shape1 = {2};

    output_shape = AddVisitor::calc_output_shape(shape1, shape2);
    REQUIRE(output_shape == shape2);
    output_shape = AddVisitor::calc_output_shape(shape2, shape1);
    REQUIRE(output_shape == shape2);
  }

  SECTION("scalar") {
    std::vector<int> shape1 = {1};
    
    output_shape = AddVisitor::calc_output_shape(shape1, shape2);
    REQUIRE(output_shape == shape2);
    output_shape = AddVisitor::calc_output_shape(shape2, shape1);
    REQUIRE(output_shape == shape2);
  }

  SECTION("error handling") {
    std::vector<int> wrong_shp = {2, 2};

    REQUIRE_THROWS(AddVisitor::calc_output_shape(wrong_shp, shape2));
    REQUIRE_THROWS(AddVisitor::calc_output_shape(shape2, wrong_shp));
  }
}

// TEST_CASE("add visitor tests", "[add][visit]") {
//   using namespace abyss::core;
//   ArrayDesc desc1{0, {3}, {1}};
//   ArrayDesc desc2{0, {3}, {1}};

//   ArrayImpl<int32_t> arr1 = {1, 2, 3};
//   ArrayImpl<int32_t> arr2 = {1, 2, 3};
//   ArrayImpl<int32_t>* result;

//   AddVisitor vis(desc1, desc2);
//   vis.visit(&arr1, &arr2);
//   // result = dynamic_cast<ArrayImpl<int32_t>*>(vis.result().data());
//   result = dynamic_cast<ArrayImpl<int32_t>*>(vis.data());

//   REQUIRE(result->size() == 3);
//   for (size_t i = 0; i < 3; i++) {
//     INFO(i);
//     CHECK(result->at(i) == 2 * (i + 1));
//   }

// }

// TEST_CASE("new scalar type", "[.][x]") {
//   using namespace abyss;

//   x::ScalarType dtype = x::kInt32;
//   core::Dispatcher<x::ScalarType> tp(dtype);
//   core::EmptyVisitor vis({3, 2});

//   tp.accept(&vis);

//   Tensor result = vis;
//   CHECK(result.shape() == std::vector<int>{3, 2});
// }