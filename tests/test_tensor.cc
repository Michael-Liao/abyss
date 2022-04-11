#include <catch2/catch.hpp>
#include <sstream>

#include "core/visitor.h"
#include "functional.h"
#include "operators.h"
#include "tensor.h"
#include "types.h"
// #include "traits.h"

TEST_CASE("test Tensor factory: empty", "[Tensor][factory][empty]") {
  abyss::Tensor a = abyss::empty({3, 2});

  REQUIRE(a.shape() == std::vector<int>{3, 2});
  REQUIRE(a.strides() == std::vector<int>{2, 1});
  REQUIRE(a.size() == 6);

  // dtype defaults to float64
  REQUIRE(a.dtype() == abyss::kFloat64);
  REQUIRE_FALSE(a.dtype() == abyss::kInt32);
  REQUIRE_FALSE(a.data() == nullptr);
}

// TEST_CASE("test array construction", "[Tensor][array]") {
//   // abyss::Tensor a = {1, 2, 3}; // won't compile
//   SECTION("direct construction from Tensor::Initializer object") {
//     abyss::Tensor a = abyss::Tensor::Initializer<int>({{1, 2, 3}, {1, 2,
//     3}});

//     REQUIRE(a.shape() == std::vector<int>{2, 3});
//     REQUIRE(a.dtype() == abyss::kInt32);
//   }

//   SECTION("using convience function") {
//     abyss::Tensor a = abyss::array<double>({{1, 2, 3}, {1, 2, 3}});
//     REQUIRE(a.shape() == std::vector<int>{2, 3});
//     REQUIRE(a.dtype() == abyss::kFloat64);
//   }
// }

TEST_CASE("Tensor interaction with native types", "[Tensor][scalar][constructor][conversion]") {
  
  SECTION("construction and assignment of scalar") {
    abyss::Tensor scalar = 2;

    REQUIRE(scalar.dtype() == abyss::kInt32);
    REQUIRE(scalar.size() == 1);

    scalar = 3.0;
    REQUIRE(scalar.dtype() == abyss::kFloat64);
  }

  SECTION("implicit conversion back to native types") {
    abyss::Tensor scalar = 2;

    // casting uses the implicit conversion
    REQUIRE(static_cast<int>(scalar) == 2);
    REQUIRE(2 == scalar);
    REQUIRE(scalar == 2);

    abyss::Tensor arr = abyss::full({2, 1}, 2);
    REQUIRE_NOTHROW(arr == 2); // broadcasted comparison

    bool all_true;
    REQUIRE_THROWS(all_true = arr == 2);
  }
}

TEST_CASE("test Tensor factory function: full", "[Tensor][factory][full]") {
  auto arr = abyss::full({2, 3}, 2);
  // auto arr = abyss::full({2, 3}, 2, abyss::kInt32);

  REQUIRE(arr.shape() == std::vector<int>{2, 3});
  REQUIRE(arr.strides() == std::vector<int>{3, 1});
  REQUIRE(arr.size() == 6);
  REQUIRE(arr.dtype() == abyss::kInt32);
  // REQUIRE(arr.dtype() == abyss::stypeof<int>()); // traits also work

  arr = abyss::full({3, 2}, 2.0);
  REQUIRE(arr.shape() == std::vector<int>{3, 2});
  REQUIRE(arr.strides() == std::vector<int>{2, 1});
  REQUIRE(arr.size() == 6);
  REQUIRE(arr.dtype() == abyss::kFloat64);
}

TEST_CASE("tensor copy construction/assignment", "[Tensor][constructor]") {
  using namespace abyss;
  auto original = full({2, 2}, 1);

  SECTION("copy constructor") {
    Tensor copied = original;

    REQUIRE(copied.size() == original.size());
    REQUIRE(copied.shape() == original.shape());
    REQUIRE(copied.data() == original.data());
  }
}

TEST_CASE("test Tensor factory function: arange", "[Tensor][factory][arange]") {
  auto arr = abyss::arange(6);

  REQUIRE(arr.size() == 6);
}

// TEST_CASE("test Tensor nested initialization", "[Tensor][nested]") {
//   SECTION("one layer") {
//     abyss::Tensor list = {1, 2, 3};
//     // abyss::Tensor list = {1, 2, 3};
//     REQUIRE(list.dtype() == abyss::kInt32);
//     REQUIRE(list.shape() == std::vector<int>{3});
//     REQUIRE(list.strides() == std::vector<int>{1});
//     REQUIRE(list.size() == 3);
//   }

//   SECTION("multiple layers") {
//     abyss::Tensor nested_list = {{{1, 2, 3}, {2, 1, 3}}};
//     REQUIRE(nested_list.shape() == std::vector<int>{1, 2, 3});
//     REQUIRE(nested_list.strides() == std::vector<int>{6, 3, 1});
//   }

//   // try {1, 2, 3.0}, tuples are not supported
//   // different shape
// }

TEST_CASE("tensor printing", "[Tensor][.][print]") {

  SECTION("integral types") {
    abyss::Tensor t1 = abyss::full({4, 1, 2}, 1);
    std::ostringstream oss;
    oss << t1;

    std::string result_str = "\n[[[ 1, 1]]\n [[ 1, 1]]\n [[ 1, 1]]\n [[ 1, 1]]]\n";

    CHECK(oss.str() == result_str);
  }

  SECTION("boolean type") {
    abyss::Tensor t1 = abyss::full({4, 1, 2}, true);
    std::ostringstream oss;
    oss << t1;

    std::string result_str = "\n"
    "[[[  true,  true]]\n"
    " [[  true,  true]]\n"
    " [[  true,  true]]\n"
    " [[  true,  true]]]\n";

    CHECK(oss.str() == result_str);
  }

  SECTION("floating point types") {
    auto t1 = abyss::arange(12, abyss::kFloat64).reshape({4, 1, 3});
    std::ostringstream oss;
    oss << t1;

    std::string result_str = "\n"
      "[[[  0,  1,  2]]\n"
      " [[  3,  4,  5]]\n"
      " [[  6,  7,  8]]\n"
      " [[  9, 10, 11]]]\n";

    CHECK(oss.str() == result_str);

    SECTION("actual floating") {
      oss.str("");
      abyss::Tensor scale = 1.2;
      t1 = scale * t1;

      oss << t1;

      result_str = "\n"
      "[[[    0,  1.2,  2.4]]\n"
      " [[  3.6,  4.8,    6]]\n"
      " [[  7.2,  8.4,  9.6]]\n"
      " [[ 10.8,   12, 13.2]]]\n";
    }

    CHECK(oss.str() == result_str);
  }

  // SECTION("floating point types with scientific expression") {
  //   auto t1 = abyss::arange(12, abyss::kFloat64).reshape({4, 1, 3});
  //   std::ostringstream oss;
  //   oss << t1;

  //   std::string result_str = "\n"
  //     "[[[  0.0,  1.0,  2.0]]\n"
  //     " [[  3.0,  4.0,  5.0]]\n"
  //     " [[  6.0,  7.0,  8.0]]\n"
  //     " [[  9.0, 10.0, 11.0]]]\n";

  //   CHECK(oss.str() == result_str);
  // }

}

TEST_CASE("tensor copy", "[Tensor][copy]") {
  abyss::Tensor t = abyss::full({3, 2}, 2);

  SECTION("shallow copy") {
    abyss::Tensor t1 = t;

    REQUIRE(t1.dtype() == t.dtype());
    REQUIRE(t1.shape() == t.shape());
    REQUIRE(t1.data() == t.data());
  }

  SECTION("deep copy") {
    abyss::Tensor t2 = t.copy();

    REQUIRE(t2.dtype() == t.dtype());
    REQUIRE(t2.shape() == t.shape());
    REQUIRE_FALSE(t2.data() == t.data());
  }
}

TEST_CASE("tensor concatenation", "[Tensor][concat]") {
  abyss::Tensor t1 = abyss::full({3, 2}, 11);

  SECTION("default operation") {
    abyss::Tensor t2 = abyss::full({3, 2}, 2);

    auto t = abyss::concat({t1, t2}, /*axis=*/0);

    REQUIRE(t.shape() == std::vector<int>{6, 2});
    REQUIRE(t.dtype() == abyss::kInt32);
    // std::cout<< t <<std::endl;

    t = abyss::concat({t1, t2}, /*axis=*/1);
    REQUIRE(t.shape() == std::vector<int>{3, 4});
    // std::cout<< t <<std::endl;
  }

  SECTION("different shape") {
    abyss::Tensor t2 = abyss::full({3, 1}, 1);

    auto t = abyss::concat({t1, t2}, /*axis=*/1);

    REQUIRE(t.shape() == std::vector<int>{3, 3});
  }

  SECTION("bad shape") {
    auto t2 = abyss::full({2, 3}, 0);

    REQUIRE_THROWS(abyss::concat({t1, t2}));
  }

  SECTION("multiple tensors") {
    std::vector<abyss::Tensor> tensors(3);
    for (auto&& t : tensors) {
      t = abyss::full({3, 2}, 2);
    }

    // std::cout<< tensors[0] <<std::endl;

    auto t_res = abyss::concat(tensors);

    REQUIRE(t_res.shape() == std::vector<int>{9, 2});
  }

  SECTION("multiple dimensions") {
    std::vector<abyss::Tensor> tensors(3, 0);
    for (auto&& t : tensors) {
      t = abyss::full({3, 2, 4}, 2);
    }

    auto t_res = abyss::concat(tensors, /*axis=*/1);

    REQUIRE(t_res.shape() == std::vector<int>{3, 6, 4});

    // std::cout<<t_res<<std::endl;
  }
}

TEST_CASE("Tensor all operation", "[Tensor][all]") {
  auto tensor = abyss::full({3, 2}, true);

  SECTION("implicit conversion") {
    bool all_true = tensor.all();

    REQUIRE(all_true);
  }

  SECTION("reduction of specific axis") {
    auto tsr1 = tensor.all(/*axis=*/0);

    REQUIRE(tsr1.shape() == std::vector<int>{1, 2});
    REQUIRE(tsr1.strides() == std::vector<int>{2, 1});
    REQUIRE(tsr1.dtype() == abyss::kBool);

    tsr1 = tensor.all(/*axis=*/1);
    REQUIRE(tsr1.shape() == std::vector<int>{3, 1});
    REQUIRE(tsr1.strides() == std::vector<int>{1, 1});
    REQUIRE(tsr1.dtype() == abyss::kBool);
  }
}

TEST_CASE("tensor comparison equal", "[Tensor][comparison][equal]") {
  auto tensor = abyss::full({3, 2}, 1);

  SECTION("should produce bool tensors") {
    auto tensor1 = abyss::full({3, 2}, 1);
    auto result = tensor == tensor1;

    REQUIRE(result.shape() == std::vector<int>{3, 2});
    REQUIRE(result.dtype() == abyss::kBool);
  }
}