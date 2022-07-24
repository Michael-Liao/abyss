#include "nn/module.h"

// #include "functional.h"
#include "operators.h"

namespace abyss::nn {
std::vector<Tensor> Module::parameters_;
std::unordered_map<std::string, Tensor*> Module::states_;

Tensor& make_parameter(Tensor data, bool requires_grad) {
  data.set_flag(abyss::core::FlagId::kRequiresGrad, requires_grad);
  Module::parameters_.emplace_back(data);

  return Module::parameters_.back();
}

std::vector<Tensor>& Module::parameters() { return parameters_; }

// Tensor Module::operator()(Tensor input_batch) {
//   Tensor out;
//   // loop through batch
//   /// @todo threading
//   for (size_t i = 0; i < input_batch.shape(0); i++) {
//     Tensor tmp = forward(input_batch(i));
//     if (i == 0) {
//       out = tmp;
//     } else {
//       out = concat({out, tmp});
//     }
//   }

//   return out;
// }

Linear::Linear(int in_features, int out_features, bool bias) {
  weight_ = make_parameter(randn({out_features, in_features}));
  bias_ = full({in_features, 1}, 0.0);
  if (bias) {
    bias_ = make_parameter(bias_, true);
  }
}

Tensor Linear::forward(Tensor input) { return matmul(weight_, input) + bias_; }

}  // namespace abyss::nn
