#include "optimizers.h"

#include "functional.h"
#include "operators.h"

namespace abyss::optim {

Optimizer::Optimizer(std::vector<Tensor>& parameters) : params_{parameters} {}

void Optimizer::zero_grad() {
  for (auto&& p : params_) {
    p.grad() = full(p.shape(), 0, p.dtype());
  }
}

SGD::SGD(std::vector<Tensor>& parameters, double lr)
    : Optimizer{parameters}, learning_rate_{lr} {}

void SGD::step() {
  /// @todo threading (or CUDA stream)
  for (auto& p : params_) {
    p = p - learning_rate_ * p.grad();
  }
}
}  // namespace abyss::optim
