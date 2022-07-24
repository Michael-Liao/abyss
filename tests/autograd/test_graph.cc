#include <bitset>
#include <catch2/catch.hpp>
#include <vector>

// #include "tensor.h"
#include "autograd/function.h"
#include "autograd/graph.h"
#include "core/utility.h"
#include "functional.h"

TEST_CASE("test tensor construction", "[Tensor]") {
  auto x = abyss::full({3, 2}, 2);

  REQUIRE_FALSE(x.flags(abyss::core::FlagId::kRequiresGrad));
  REQUIRE(x.shape() == std::vector<int>{3, 2});
}

TEST_CASE("what happens to gradients and gradient functions", "[Tensor]") {
  auto x = abyss::full({3, 2}, 2);

  REQUIRE_THROWS(x.grad());
}

TEST_CASE("check hash", "[Tensor][hash]") {
  using namespace abyss;
  auto a = full({3, 2}, 1);
  size_t hash1 = std::hash<Tensor>()(a);
  size_t hash2 = 0;

  SECTION("copied tensor must have same hash") {
    auto b = a;
    hash2 = std::hash<Tensor>()(b);

    REQUIRE(hash1 == hash2);
  }

  SECTION("requires_grad changes hash") {
    auto b = a;
    b.set_flag(core::FlagId::kRequiresGrad, true);
    hash2 = std::hash<Tensor>()(b);

    REQUIRE_FALSE(hash1 == hash2);
  }

  SECTION("deep copy changes hash") {
    auto b = a.copy();
    b.set_flag(core::FlagId::kRequiresGrad, true);
    hash2 = std::hash<Tensor>()(b);

    REQUIRE_FALSE(hash1 == hash2);
  }
}

TEST_CASE("test tensor backprop", "[Tensor][backprop][add]") {
  using namespace abyss;

  auto a = full({3, 2}, 2);
  a.set_flag(core::FlagId::kRequiresGrad, true);
  auto b = full({3, 2}, 2);
  b.set_flag(core::FlagId::kRequiresGrad, true);

  auto fwd_result = full({3, 2}, 4);
  // std::cout<< "?" << a <<std::endl;

  auto c = add(a, b);
  bool fwd_true = (c == fwd_result).all();
  REQUIRE(fwd_true);

  REQUIRE(c.flags(core::FlagId::kRequiresGrad));
  // REQUIRE_NOTHROW(c.grad());
  REQUIRE_NOTHROW(c.grad_fn());

  REQUIRE_FALSE(autograd::Graph::instance().edges().empty());

  auto grad = full({3, 2}, 1);
  c.backward(grad);

  // clears graph after wew finish back prop
  CHECK(autograd::Graph::instance().edges().empty());

  // std::cout<< a.grad() << std::endl;
  // INFO(a.grad());
  bool all_true = (a.grad() == 1).all();
  REQUIRE(all_true);
  all_true = (b.grad() == 1).all();
  REQUIRE(all_true);
}

TEST_CASE("matmul backprop", "[Tensor][backprop][matmul]") {
  using namespace abyss;

  auto w = full({3, 2}, 2);
  auto x = full({2, 1}, 1);
  w.set_flag(core::FlagId::kRequiresGrad, true);
  x.set_flag(core::FlagId::kRequiresGrad, true);

  bool grad_init_zeros = (w.grad() == 0).all();
  REQUIRE(grad_init_zeros);

  auto y = matmul(w, x);

  y.backward(full({3, 1}, 1));

  REQUIRE_NOTHROW(w.grad());
  REQUIRE_NOTHROW(x.grad());

  // std::cout << w.grad() << std::endl;

  // CHECK(w.grad());
  bool all_true = (w.grad() == 1).all();
  REQUIRE(all_true);

  // std::cout<< x.grad() << std::endl;

  all_true = (x.grad() == 6).all();
  REQUIRE(all_true);
}

TEST_CASE("conjunction of operations", "[conj][backprop]") {
  using namespace abyss;
  auto w = full({3, 2}, 2);
  w.set_flag(core::FlagId::kRequiresGrad, true);

  SECTION("normal operation") {
    auto x = full({2, 1}, 1);
    x.set_flag(core::FlagId::kRequiresGrad, true);

    auto b = full({3, 1}, 1);
    b.set_flag(core::FlagId::kRequiresGrad, true);

    auto y = matmul(w, x) + b;

    REQUIRE(y.flags(core::FlagId::kRequiresGrad) == true);

    y.backward(full({3, 1}, 1));

    CHECK(w.grad().shape() == std::vector<int>{3, 2});

    bool all_true = (x.grad() == abyss::full({2, 1}, 6.0)).all();
    CHECK(all_true);
  }

  SECTION("requires correct broadcast") {
    auto x = full({2}, 1); // broadcast required
    x.set_flag(core::FlagId::kRequiresGrad, true);
  }
}