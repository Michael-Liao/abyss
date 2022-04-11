#include "catch2/catch.hpp"

#include "core/ndarray.h"

TEST_CASE("ndarray interface", "[NDArray][constructor]") {
  using namespace abyss::core;

  NDArray<int> arr = 1;
  SECTION("constructing scalar") {
    REQUIRE(arr[0] == 1);
  }

  SECTION("constructing through fill value") {
    arr = NDArray<int>({2, 3}, 3);
    REQUIRE(arr.shape() == std::vector<int>{2, 3});
    REQUIRE(arr.strides() == std::vector<int>{3, 1});
    REQUIRE(arr.size() == 6);
    REQUIRE(arr.nbytes() == 6 * sizeof(int));

    for (size_t i = 0; i < arr.size(); i++) {
      REQUIRE(arr[i] == 3);
    }
  }

  SECTION("constructing from vector with shape") {
    std::vector<int> values = {1, 2, 3, 4, 5, 6};
    arr = NDArray<int>({2, 3}, values.begin(), values.end());

    for (size_t i = 0; i < values.size(); i++) {
      REQUIRE(arr[i] == values[i]);
    }

    SECTION("wrong shaped vector") {
      values = {1, 2, 3, 4};
      REQUIRE_THROWS(NDArray<int>({2, 3}, values.begin(), values.end()));
    }
  }
}

TEST_CASE("NDArray iterator", "[NDArray][Iterator]") {
  using namespace abyss::core;

  NDArray<int> arr({4, 2, 3});

  SECTION("iterator returns slices") {
    auto it = arr.begin();

    for (int i = 0; i < 4; i++) {
      auto slice = *it; // ndarray
      
      REQUIRE(std::is_same<decltype(slice), NDArray<int>>::value);
      REQUIRE(slice.is_view() == true);
      REQUIRE(slice.shape() == std::vector<int>{2, 3});
      REQUIRE(slice.strides() == std::vector<int>{3, 1});

      it++;
    }

    REQUIRE(it == arr.end());
    REQUIRE(it - arr.begin() == 4);
  }

  SECTION("slices of slices") {
    // equivilent to `arr[0, 1, :]` in numpy
    auto vec = *((*arr.begin()).begin() + 1);

    REQUIRE(std::is_same<decltype(vec), NDArray<int>>::value);
    REQUIRE(vec.shape() == std::vector<int>{3});

    auto scalar = *vec.begin();
    REQUIRE(scalar.shape() == std::vector<int>{1});
  }

  SECTION("undefined behavior accessing views outside of scope") {
    NDArray<int> view;
    
    {
      NDArray<int> arr1({3, 2}, 10);
      REQUIRE(arr1.is_view() == false);

      view = *(arr1.begin() + 1);
    }

    REQUIRE(view.is_view() == true);
    CHECK(view.data()[0] == 10);    
  }

  SECTION("equality of ndarray iterators") {
    auto it1 = arr.begin();
    auto it2 = arr.begin();

    REQUIRE(it1 == it2);
    REQUIRE(it1 + 1 == it2 + 1);
    it1++;
    ++it1;
    REQUIRE(it1 == it2 + 2);
    REQUIRE(it1 - 2 == arr.begin());
    REQUIRE(it1 - arr.begin() == 2);
    it1--;
    --it1;
    REQUIRE(it1 == arr.begin());
  }

  SECTION("ndarray values are assignable") {
    NDArray<int> arr1({3, 2}, 1);

    // view has to take reference or else it will
    // result into a copy, which will not be the same object anymore
    auto& view = *(arr1.begin() + 1);

    view[1] = 2;
    REQUIRE(view[0] == 1);
    REQUIRE(view[1] == 2);

    // REQUIRE(arr1.begin() + 1 == view.begin());
    INFO(arr1.data() - view.data());
    REQUIRE(arr1.data() + 2 == view.data());
    REQUIRE(arr1[3] == 2);
  }

  SECTION("ndarray reverse iterator") {
    std::initializer_list<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    NDArray<int> arr1({2, 2, 3}, data.begin(), data.end());

    auto& rview = *arr1.rbegin();
    REQUIRE(rview[0] == 7);

    auto it = arr1.rbegin();
    for (size_t i = 0; i < 6; i++) {
      REQUIRE((*it)[i] == 7 + i);
    }
    
    auto it1 = it + 1;
    REQUIRE(it1 == ++it);
    REQUIRE(it1 - 1 == arr1.rbegin());
    // slightly not inline with the standard
    // but the iterator is non-standard anyway
    REQUIRE(arr1.rbegin()->data() == (arr1.end() - 1)->data());
    REQUIRE(arr1.rend() - arr1.rbegin() == 2);
  }
}