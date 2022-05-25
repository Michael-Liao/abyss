#include <catch2/catch.hpp>
#include <memory>

#include "core/array.h"
#include "core/utility.h"
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

TEST_CASE("n-dim iterators that calculates proper offsets", "[array][NDIterator]") {
  using namespace abyss::core;
  auto arr = ArrayImpl<int>::from_range(6);

  SECTION("contiguous array") {
    ArrayDesc desc{0, {2, 3}, {3, 1}};

    auto it = arr.nbegin(desc);
    for (size_t i = 0; i < 2 * 3; i++) {
      REQUIRE(*it == i);
      it++;
    }
    
    REQUIRE(it == arr.nend(desc));
  }

  SECTION("a slice") {
    ArrayDesc desc{3, {3}, {1}};

    auto it = arr.nbegin(desc);
    int count = 0;
    while (it != arr.nend(desc)) {
      ++it;
      ++count;
    }

    REQUIRE(count == it - arr.nbegin(desc));
    REQUIRE_FALSE(it - arr.nbegin(desc) == arr.size());
  }
}
