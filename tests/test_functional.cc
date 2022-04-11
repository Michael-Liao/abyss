#include <catch2/catch.hpp>

#include <vector>

#include "functional.h"
#include "types.h"

TEST_CASE("test add function on scalars (int, double)", "[functional][add][scalar][int-double]") {
  abyss::Tensor a = 1;
  abyss::Tensor b = 2.0;

  abyss::Tensor c = abyss::add(a, b);
  
  REQUIRE(c.dtype() == abyss::kFloat64);
  REQUIRE(c.size() == 1);
  REQUIRE(c.shape() == std::vector<int>{1});
  REQUIRE(c.strides() == std::vector<int>{1});

  // std::cout<< c <<std::endl;
}

TEST_CASE("tensor addition", "[functional][add]") {
  auto a = abyss::full({3, 2}, 1);
  
  SECTION("basic case with same shapes"){
    auto b = abyss::full({3, 2}, 2);

    auto c = abyss::add(a, b);

    REQUIRE(c.dtype() == abyss::kInt32);
    REQUIRE(c.shape() == std::vector<int>{3, 2});
    REQUIRE(c.strides() == std::vector<int>{2, 1});
  }

  SECTION("broadcast rule 1: promote shape with one") {
    auto b1 = abyss::full({3, 1}, 2);

    auto c = abyss::add(a, b1);
    REQUIRE(c.shape() == std::vector<int>{3, 2});
  }
}

TEST_CASE("tensor subtraction", "[functional][subtract]") {
  auto a = abyss::full({3, 2}, 1);

  SECTION("basic case with same shape") {
    auto b = abyss::full({3, 2}, 2);

    auto c = abyss::subtract(a, b);
    REQUIRE(c.shape() == std::vector<int>{3, 2});
  }
}

// TEST_CASE("test matmul function (int, float)", "[functions][matmul][int-float]") {
//   abyss::Tensor a = {1, 2, 3};
//   abyss::Tensor b = abyss::full({3}, 2.0f);

//   REQUIRE(a.shape() == std::vector<int>{3});

//   auto c = abyss::matmul(a, b);

//   REQUIRE(c.shape() == std::vector<int>{1});
// }