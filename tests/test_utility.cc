#include <catch2/catch.hpp>

#include "core/array.h"
#include "core/utility.h"

TEST_CASE("test unravel index", "[core][utility][unravel_index]") {
  using namespace abyss;
  std::vector<int> coords;
  std::vector<int> tgt_coords;

  SECTION("normal situation") {
    coords = core::unravel_index(1, {2, 2});
    tgt_coords = {0, 1};
    REQUIRE(coords.size() == 2);
    REQUIRE(coords == tgt_coords);
  }

  SECTION("throws exception when exceeding the size of array") {
    REQUIRE_THROWS(core::unravel_index(10, {2, 3}));
  }

  // SECTION("?x") {
  //   coords = core::unravel_index(10, {2, 3});
  //   tgt_coords = {0, 0};
  //   REQUIRE(coords == tgt_coords);
  //   // INFO(coords);
  // }
}

TEST_CASE("test utility functions is broadcastable", "[utility][is_broadcastable]") {
  using namespace abyss;
  std::vector<int> shp1 = {4, 2, 3};
    std::vector<int> shp2 = shp1;
  
  SECTION("broadcastable with same shape") {
    REQUIRE(core::is_broadcastable(shp1, shp2));
  }

  SECTION("broadcastable with same dimensions") {
    shp2 = {1, 2, 3};
    REQUIRE(core::is_broadcastable(shp1, shp2));

    shp2 = {4, 2, 1};
    REQUIRE(core::is_broadcastable(shp1, shp2));

    shp2 = {4, 1, 1};
    REQUIRE(core::is_broadcastable(shp1, shp2));
  }

  SECTION("broadcastable with different dimensions") {
    shp2 = {1};
    REQUIRE(core::is_broadcastable(shp1, shp2));

    shp2 = {1, 3};
    REQUIRE(core::is_broadcastable(shp1, shp2));

    shp2 = {3};
    REQUIRE(core::is_broadcastable(shp1, shp2));
  }

  SECTION("wrong shapes") {
    shp2 = {2, 2, 3};
    REQUIRE_FALSE(core::is_broadcastable(shp1, shp2));

    shp2 = {3, 3};
    REQUIRE_FALSE(core::is_broadcastable(shp1, shp2));
  }
}

TEST_CASE("copy", "[core][utility][copy]") {
  using namespace abyss::core;

  ArrayDesc src_desc{0, {4, 1, 2}, {2, 2, 1}};
  std::vector<int> src_data = {0, 1, 2, 3, 4, 5, 6, 7};

  SECTION("copy contiguous into contiguous") {
    std::vector<int> dst_data(src_data.size(), -1);
    auto end_it = copy(src_data.begin(), src_data.end(), src_desc, dst_data.begin(), src_desc);

    REQUIRE(end_it == dst_data.end());
    REQUIRE(src_data == dst_data);
  }

  SECTION("copy slice into continuous") {
    // sliced from {4, 2, 2}
    src_desc.strides = {4, 2, 1};
    ArrayDesc dst_desc{0, {4, 1, 2}, {2, 2, 1}};
    src_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<int> tgt_data = {0, 1, 4, 5, 8, 9, 12, 13};

    std::vector<int> dst_data(tgt_data.size(), -1);

    auto end_it = copy(src_data.begin(), src_data.end(), src_desc, dst_data.begin(), dst_desc);
    
    REQUIRE(end_it - dst_data.end() == 0);
    CHECK(dst_data == tgt_data);
  }

  SECTION("copy slice into slice") {
    src_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<int> dst_data = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

    // sliced from 4, 2, 2
    ArrayDesc desc{0, {4, 1, 2}, {4, 2, 1}};

    copy(src_data.begin(), src_data.end(), desc, dst_data.begin(), desc);

    std::vector<int> tgt_data = {0, 1, 13, 12, 4, 5, 9, 8, 8, 9, 5, 4, 12, 13, 1, 0};
    CHECK(dst_data == tgt_data);
  }

  SECTION("copy contiguous into transposed") {
    /// @note transpose should not be copied into
    // ArrayDesc dst_desc{0, {2, 1, 4}, {1, 2, 2}}; // transposed
    // std::vector<int> dst_data(src_data.size(), -1);

    // std::vector<int> tgt_data = {0, 2, 4, 6, 1, 3, 5, 7};

    // REQUIRE_FALSE(dst_data == src_data);

    // auto end_it = copy(src_data.begin(), src_data.end(), src_desc, dst_data.begin(), dst_desc);

    // REQUIRE(end_it - dst_data.end() == 0);
    // CHECK(dst_data == tgt_data);
  }
}


TEST_CASE("broadcast copy", "[core][utility][broadcast_copy]") {
  using namespace abyss::core;

  SECTION("broadcast with same dimension") {
    ArrayDesc src_desc{0, {4, 1, 2}, {2, 2, 1}};
    // std::vector<int> shape = {4, 1, 2};
    // auto arr = ArrayImpl<int32_t>::from_range(8);
    ArrayDesc dst_desc{0, {4, 3, 2}, {6, 2, 1}};
    // std::vector<int> dst_shape = {4, 3, 2};
    // auto arr_out = std::make_shared<ArrayImpl<int32_t>>(24);

    std::vector<int32_t> target = {0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3,
                                   4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7};

    SECTION("using std::vector data") {
      std::vector<int> arr = {0, 1, 2, 3, 4, 5, 6, 7};
      std::vector<int> arr_out(24, -1);
      broadcast_copy(arr.begin(), arr.end(), src_desc, arr_out.begin(), dst_desc);

      for (size_t i = 0; i < 24; i++) {
        INFO("index: " << i);
        CHECK(arr_out.at(i) == target[i]);
      }
    }

    SECTION("using ArrayImpl data") {
      auto arr = ArrayImpl<int32_t>::from_range(8);
      auto arr_out = ArrayImpl<int32_t>(24, -1);
      broadcast_copy(arr.begin(), arr.end(), src_desc, arr_out.begin(), dst_desc);

      for (size_t i = 0; i < 24; i++) {
        INFO("index: " << i);
        CHECK(arr_out.at(i) == target[i]);
      }
    }
  }

  SECTION("broadcast to higher dimensions") {
    // std::vector<int> d_shape = {4, 3};
    ArrayDesc dst_desc;
    dst_desc.shape = {4, 3};
    dst_desc.strides = shape2strides(dst_desc.shape);
    auto arr = ArrayImpl<int32_t>::from_range(3);

    std::vector<int> target = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};

    SECTION("normal case") {
      // std::vector<int> shape = {3};
      ArrayDesc src_desc;
      src_desc.shape = {3};
      src_desc.strides = {1};

      ArrayImpl<int32_t> out(12);
      broadcast_copy(arr.begin(), arr.end(), src_desc, out.begin(), dst_desc);

      for (size_t i = 0; i < 12; i++) {
        INFO("index: " << i);
        CHECK(out[i] == target[i]);
      }
    }

    SECTION("including leading shape of 1") {
      std::vector<int> shape = {1, 3};
      ArrayDesc src_desc{0, {1, 3}, {3, 1}};
      ArrayImpl<int32_t> out(12);
      
      broadcast_copy(arr.begin(), arr.end(), src_desc, out.begin(), dst_desc);

      for (size_t i = 0; i < 12; i++) {
        INFO("index: " << i);
        CHECK(out[i] == target[i]);
      }
    }
  }

  SECTION("fit scalar to tensor") {
    std::vector<int> shape = {1};
    ArrayDesc src_desc{0, {1}, {1}};
    std::vector<int> d_shape = {4, 2, 2};
    ArrayDesc dst_desc{0, {4, 2, 2}, shape2strides({4, 2, 2})};

    ArrayImpl<int32_t> arr(1, 1); // scalar
    ArrayImpl<int32_t> out(16);

    broadcast_copy(arr.begin(), arr.end(), src_desc, out.begin(), dst_desc);

    for (size_t i = 0; i < 16; i++) {
      CHECK(out[i] == 1);
    }
  }

  // SECTION("multiple broadcasts") {
  //   // ultimate test with all above conditions
  //   // Input:      1 x 3 x 1 x 2
  //   // Output: 2 x 4 x 3 x 2 x 2
  //   //         ^~~~~~~~~~~~~~~~~~~ empty dimension
  //   //             ^~~~~~~~~~~~~~~ leading 1
  //   //                     ^~~~~~~ in-shape broadcast
  //   std::vector<int> shape = {1, 3, 1, 2};
  //   std::vector<int> d_shape = {2, 4, 3, 2, 2};

  //   ArrayImpl<int32_t> arr(6, 2);
  //   ArrayImpl<int32_t> out(96, -1);

  //   ArrayImpl<int32_t>::iterator end_it;

  //   end_it = broadcast_copy(arr.begin(), arr.end(), shape, out.begin(), d_shape);

  //   for (auto &&v : out) {
  //     REQUIRE(v == 2);
  //   }

  //   REQUIRE(end_it == out.end());
  // }

  // SECTION("ignore dimemsion") {
  //   // this is for matmul broadcast schemes
  //   // shape1: 4 x 9 x 3 x 2
  //   // shape2:         1 x 2 x 5
  //   // output: 4 x 9 x 3 x 5
  // }
}