#include <catch2/catch.hpp>

#include "tensor.h"
#include "operators.h"
#include "nn/losses.h"

TEST_CASE("negative log likelihood loss", "[loss][nll]") {
  abyss::nn::NLLLoss loss_fn;

  auto input = abyss::full({8, 3}, 0.0);
  auto target = abyss::full({8}, 0);
  input(0, 0) = -0.1;
  input(0, 1) = -0.2;
  input(0, 2) = -0.7;

  target(1) = 2;

  // std::cout<< input << std::endl;
  // std::cout<< target << std::endl;

  auto loss = loss_fn(input, target);

  REQUIRE(loss.shape() == std::vector<int>{1});
  REQUIRE(loss.strides() == std::vector<int>{1});

  // std::cout<< loss << std::endl;

  bool ok = (loss == 0.0875).all();
  REQUIRE(ok);
}