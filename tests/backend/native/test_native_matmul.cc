#include <vector>

#include "backend/matmul.h"
#include "catch2/catch.hpp"

TEST_CASE("matmul with blas", "[native][matmul]") {
  using namespace abyss::backend;
  // std::vector<int> I = {1, 0, 0, 1};

  SECTION("square to square") {
    std::vector<int> I = {1, 0, 0, 1};
    std::vector<int> X = {1, 2, 3, 4};
    std::vector<int> Y(4);

    matmul(I.data(), X.data(), 2, 2, 2, Y.data());

    for (size_t i = 0; i < X.size(); i++) {
      REQUIRE(Y[i] == X[i]);
    }
  }

  SECTION("square to vector") {
    std::vector<int> I = {1, 0, 0, 1};
    std::vector<int> X = {1, 2};
    std::vector<int> Y(4);

    matmul(I.data(), X.data(), 2, 1, 2, Y.data());

    for (size_t i = 0; i < X.size(); i++) {
      REQUIRE(Y[i] == X[i]);
    }
  }

  SECTION("arbitrary square to square") {
    std::vector<int> W = {1, 2, 3, 4};
    std::vector<int> X = {1, 2, 3, 4};
    std::vector<int> Y(4);

    std::vector<int> target = {1 * 1 + 2 * 3, 1 * 2 + 2 * 4, 3 * 1 + 4 * 3,
                               3 * 2 + 4 * 4};

    matmul(W.data(), X.data(), 2, 2, 2, Y.data());

    for (size_t i = 0; i < 4; i++) {
      REQUIRE(target[i] == Y[i]);
    }
  }

  SECTION("arbitrary matrix shapes") {
    std::vector<int32_t> W = {1, 2, 3, 4, 5, 6};        // 3 x 2
    std::vector<int32_t> X = {1, 2, 3, 4, 5, 6, 7, 8};  // 2 x 4

    std::vector<int32_t> target = {
        1 * 1 + 2 * 5, 1 * 2 + 2 * 6, 1 * 3 + 2 * 7, 1 * 4 + 2 * 8,
        3 * 1 + 4 * 5, 3 * 2 + 4 * 6, 3 * 3 + 4 * 7, 3 * 4 + 4 * 8,
        5 * 1 + 6 * 5, 5 * 2 + 6 * 6, 5 * 3 + 6 * 7, 5 * 4 + 6 * 8,
    };  // 3 x 4
    std::vector<int32_t> Y(target.size());

    // REQUIRE(target.size() == 12);

    matmul(W.data(), X.data(), 3, 2, 4, Y.data());

    REQUIRE(target == Y);
  }
}
