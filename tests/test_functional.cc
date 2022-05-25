#include <catch2/catch.hpp>

#include <vector>

#include "functional.h"
#include "operators.h"
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
    // limitations of c++, it is not possible to do in-place conversion.
    // The catch library is trying to convert const Tensor& into bool,
    // which does not follow the rules of conversion operator.
    // see examples: https://en.cppreference.com/w/cpp/language/cast_operator
    bool all_true = (c == abyss::full({3, 2}, -1)).all();
    REQUIRE(all_true);
    // REQUIRE((bool)(c == abyss::full({3, 2}, -1)).all());
    // REQUIRE( static_cast<bool>((c == abyss::full({3, 2}, -1)).all()) );
  }
}

TEST_CASE("test matmul function", "[functions][matmul]") {
  SECTION("interaction between matrices") {
    auto a = abyss::full({3, 2}, 1);
    auto b = abyss::full({2, 3}, 1);

    auto result = abyss::full({3, 3}, 2);

    auto c = abyss::matmul(a, b);

    REQUIRE(c.shape() == std::vector<int>{3, 3});
    bool all_true =  (c == result).all();
    CHECK(all_true);
    // std::cout<< "tensor" <<(c == result).all() << std::endl;
  }

  SECTION("interaction matrix and vector") {
    auto w = abyss::full({3, 3}, 1);
    auto x = abyss::full({3}, 1);

    auto result = abyss::full({3, 1}, 3);

    auto y = abyss::matmul(w, x);

    REQUIRE(y.shape() == std::vector<int>{3, 1});
    bool all_true = (y == result).all();
    CHECK(all_true);
  }

  SECTION("interaction between vector and vector") {
    auto x = abyss::full({3}, 1);
    auto y = abyss::full({3}, 1);

    abyss::Tensor result = 3;

    auto z = abyss::matmul(x, y);

    REQUIRE(z.shape() == std::vector<int>{1});

    CHECK(z == result);
    CHECK(z == 3);
  }

  SECTION("leading dimensions") {
    auto x = abyss::full({1, 3}, 1);
    auto y = abyss::full({3, 2}, 1);

    auto z = abyss::matmul(x, y);
    REQUIRE(z.shape() == std::vector<int>{1, 2});
  }
}