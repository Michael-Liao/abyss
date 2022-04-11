#include "catch2/catch.hpp"

#include "backend/arithmetics.h"

TEST_CASE("add function with c-like arrays", "[native][add][c_like]") {
  int a[3] = {1, 2, 3};
  int b[3] = {1, 2, 3};
  int c[3];

  abyss::backend::add(a, b, 3, c);

  for (size_t i = 0; i < 3; i++) {
    REQUIRE(a[i] == i + 1);
    REQUIRE(b[i] == i + 1);

    REQUIRE(c[i] == 2*(i + 1));
  }
}

TEST_CASE("add function with containers", "[native][add][container]") {
  std::vector<int32_t> a = {1, 2, 3};
  std::vector<int32_t> b = {1, 2, 3};
  std::vector<int32_t> c(3); // element size: 3

  abyss::backend::add(a.data(), b.data(), 3, c.data());

  for (size_t i = 0; i < 3; i++) {
    REQUIRE(a[i] == i + 1);
    REQUIRE(b[i] == i + 1);

    REQUIRE(c[i] == 2*(i + 1));
  }
}

TEST_CASE("add function with different types", "[native][add]") {
  int a[3] = {1, 2, 3};
  double b[3] = {1.1, 2.2, 3.3};
  double c[3];

  abyss::backend::add(a, b, 3, c);

  for (size_t i = 0; i < 3; i++) {
    REQUIRE(a[i] == i + 1);
    REQUIRE_THAT(b[i], Catch::Matchers::WithinRel(1.1 * (i + 1)));

    REQUIRE_THAT(c[i], Catch::Matchers::WithinRel(2.1 * (i + 1)));
  }
}