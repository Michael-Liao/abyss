#include <catch2/catch.hpp>
#include <memory>

#include "core/array.h"
// #include "types.h"

TEST_CASE("contruct Array from range", "[core][Array][from_range]") {
  using namespace abyss::core;

  SECTION("basic style") {
    auto arr = ArrayImpl<int32_t>::from_range(0, 9, 1);
    REQUIRE(arr.size() == 9);
    for (size_t i = 0; i < arr.size(); i++) {
      REQUIRE(arr.at(i) == i);
    }
  }

  SECTION("with different step") {
    auto arr = ArrayImpl<int32_t>::from_range(0, 9, 2);
    REQUIRE(arr.size() == 5);
    for (size_t i = 0; i < arr.size(); i++) {
      REQUIRE(arr.at(i) == i * 2);
    }
  }

  SECTION("with step and exact ends") {
    auto arr = ArrayImpl<int32_t>::from_range(0, 8, 2);
    REQUIRE(arr.size() == 4);
    for (size_t i = 0; i < arr.size(); i++) {
      REQUIRE(arr.at(i) == i * 2);
    }
  }

  SECTION("with different start") {
    auto arr = ArrayImpl<int32_t>::from_range(1, 9, 2);
    REQUIRE(arr.size() == 4);
    for (size_t i = 0; i < arr.size(); i++) {
      REQUIRE(arr.at(i) == i * 2 + 1);
    }
  }

  SECTION("simplified from") {
    auto arr = ArrayImpl<int32_t>::from_range(8);
    // auto arr = ArrayImpl<int32_t>::from_range(0, 8, 1);
    REQUIRE(arr.size() == 8);
    for (size_t i = 0; i < arr.size(); i++) {
      REQUIRE(arr[i] == i);
    }
  }
}

TEST_CASE("abyss Array iterator - NDIterator", "[core][Array][iterator]") {
  using namespace abyss::core;
  auto arr = ArrayImpl<int32_t>::from_range(10);

  SECTION("similar to a normal iterator when stride is 1") {
    auto arr_it = arr.begin();
    REQUIRE(*arr_it == 0);
    arr_it += 1;
    REQUIRE(arr_it - arr.begin() == 1);
    REQUIRE(*arr_it == 1);
    REQUIRE(*arr_it++ == 2);
    REQUIRE(*++arr_it == 3);
    REQUIRE(*arr_it-- == 2);
    REQUIRE(*--arr_it == 1);

    arr_it = arr.begin();
    for (int i = 0; i < arr.size(); i++) {
      arr_it++;
    }
    REQUIRE(arr_it == arr.end());

    auto it1 = arr.begin() + 1;
    REQUIRE(*it1 == arr.at(1));

    auto it2 = it1 + 2;
    REQUIRE(*it2 == arr.at(3));

    // range for, ultimate test
    int tgt = 0;
    for (auto &&v : arr) {
      REQUIRE(v == tgt);
      tgt++;
    }
  }
}

TEST_CASE("broadcast copy", "[core][Array][broadcast_copy]") {
  using namespace abyss::core;

  SECTION("broadcast with same dimension") {
    std::vector<int> shape = {4, 1, 2};
    // auto arr = ArrayImpl<int32_t>::from_range(8);
    std::vector<int> dst_shape = {4, 3, 2};
    // auto arr_out = std::make_shared<ArrayImpl<int32_t>>(24);

    std::vector<int32_t> target = {0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3,
                                   4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7};

    SECTION("using std::vector data") {
      std::vector<int> arr = {0, 1, 2, 3, 4, 5, 6, 7};
      std::vector<int> arr_out(24, -1);
      broadcast_copy(arr.begin(), arr.end(), shape, arr_out.begin(), dst_shape);

      for (size_t i = 0; i < 24; i++) {
        INFO("index: " << i);
        CHECK(arr_out.at(i) == target[i]);
      }
    }

    SECTION("using ArrayImpl data") {
      auto arr = ArrayImpl<int32_t>::from_range(8);
      auto arr_out = ArrayImpl<int32_t>(24, -1);
      broadcast_copy(arr.begin(), arr.end(), shape, arr_out.begin(), dst_shape);

      for (size_t i = 0; i < 24; i++) {
        INFO("index: " << i);
        CHECK(arr_out.at(i) == target[i]);
      }
    }
  }

  SECTION("broadcast to higher dimensions") {
    std::vector<int> d_shape = {4, 3};
    auto arr = ArrayImpl<int32_t>::from_range(3);

    std::vector<int> target = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};

    SECTION("normal case") {
      std::vector<int> shape = {3};
      ArrayImpl<int32_t> out(12);
      broadcast_copy(arr.begin(), arr.end(), shape, out.begin(), d_shape);

      for (size_t i = 0; i < 12; i++) {
        INFO("index: " << i);
        CHECK(out[i] == target[i]);
      }
    }

    SECTION("including leading shape of 1") {
      std::vector<int> shape = {1, 3};
      ArrayImpl<int32_t> out(12);
      
      broadcast_copy(arr.begin(), arr.end(), shape, out.begin(), d_shape);

      for (size_t i = 0; i < 12; i++) {
        INFO("index: " << i);
        CHECK(out[i] == target[i]);
      }
    }
  }

  SECTION("fit scalar to tensor") {
    std::vector<int> shape = {1};
    std::vector<int> d_shape = {4, 2, 2};

    ArrayImpl<int32_t> arr(1, 1); // scalar
    ArrayImpl<int32_t> out(16);

    broadcast_copy(arr.begin(), arr.end(), shape, out.begin(), d_shape);

    for (size_t i = 0; i < 16; i++) {
      CHECK(out[i] == 1);
    }
  }

  SECTION("multiple broadcasts") {
    // ultimate test with all above conditions
    // Input:      1 x 3 x 1 x 2
    // Output: 2 x 4 x 3 x 2 x 2
    //         ^~~~~~~~~~~~~~~~~~~ empty dimension
    //             ^~~~~~~~~~~~~~~ leading 1
    //                     ^~~~~~~ in-shape broadcast
    std::vector<int> shape = {1, 3, 1, 2};
    std::vector<int> d_shape = {2, 4, 3, 2, 2};

    ArrayImpl<int32_t> arr(6, 2);
    ArrayImpl<int32_t> out(96, -1);

    ArrayImpl<int32_t>::iterator end_it;

    end_it = broadcast_copy(arr.begin(), arr.end(), shape, out.begin(), d_shape);

    for (auto &&v : out) {
      REQUIRE(v == 2);
    }

    REQUIRE(end_it == out.end());
  }

  // SECTION("ignore dimemsion") {
  //   // this is for matmul broadcast schemes
  //   // shape1: 4 x 9 x 3 x 2
  //   // shape2:         1 x 2 x 5
  //   // output: 4 x 9 x 3 x 5
  // }
}

TEST_CASE("initialize array with containers", "[array][constructor]") {
  using namespace abyss::core;
  
  auto target = ArrayImpl<int32_t>::from_range(1, 4, 1);

  SECTION("using std::vector") {
    std::vector<int> vec = {1, 2, 3};
    ArrayImpl<int32_t> arr = vec;

    for (size_t i = 0; i < 3; i++) {
      REQUIRE(arr[i] == target[i]);
    }
  }

  SECTION("using std::initializer_list") {
    ArrayImpl<int32_t> arr = {1, 2, 3};

    for (size_t i = 0; i < 3; i++) {
      REQUIRE(arr[i] == target[i]);
    }
  }
}