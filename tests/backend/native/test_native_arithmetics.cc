#include "backend/arithmetics.h"
#include "catch2/catch.hpp"

TEST_CASE("add function with c-like arrays", "[native][add]") {
  SECTION("normal c-like array") {
    int a[3] = {1, 2, 3};
    size_t id_a[3] = {2, 1, 0};
    int b[3] = {1, 2, 3};
    size_t id_b[3] = {0, 1, 2};
    int c[3];

    // abyss::backend::add(a, b, 3, c);
    abyss::backend::add(a, id_a, b, id_b, 3u, c);

    for (size_t i = 0; i < 3; i++) {
      REQUIRE(c[i] == 4);
    }
  }

  SECTION("add function with different types") {
    int a[3] = {1, 2, 3};
    size_t id_a[3] = {0, 1, 2};
    double b[3] = {1.1, 2.2, 3.3};
    double c[3];

    abyss::backend::add(a, id_a, b, id_a, 3, c);

    for (size_t i = 0; i < 3; i++) {
      REQUIRE(a[i] == i + 1);
      REQUIRE_THAT(b[i], Catch::Matchers::WithinRel(1.1 * (i + 1)));

      REQUIRE_THAT(c[i], Catch::Matchers::WithinRel(2.1 * (i + 1)));
    }
  }
}

// TEST_CASE("add function with containers", "[native][add][container]") {
//   std::vector<int32_t> a = {1, 2, 3};
//   std::vector<int32_t> b = {1, 2, 3};
//   std::vector<int32_t> c(3); // element size: 3

//   abyss::backend::add(a.data(), b.data(), 3, c.data());

//   for (size_t i = 0; i < 3; i++) {
//     REQUIRE(a[i] == i + 1);
//     REQUIRE(b[i] == i + 1);

//     REQUIRE(c[i] == 2*(i + 1));
//   }
// }
