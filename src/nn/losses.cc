#include "nn/losses.h"

#include <numeric>

#include "functional.h"
#include "operators.h"

namespace abyss::nn {

// Tensor CrossEntropyLoss::forward(Tensor input, Tensor target) {
//   // shape: (batch_size, classes)
//   auto y_pred = exp(input);
//   auto y_true = exp(target);

//   auto class_loss = - y_pred / sum(y_true, 1);
  
//   // auto class_loss =
//   //     -sum(target * log(input + std::numeric_limits<double>::epsilon()), 1);
//   return sum(class_loss, 0) / input.shape(0);
// }

Tensor NLLLoss::forward(Tensor input, Tensor target) {
  if (input.shape(0) != target.shape(0)) {
    throw std::runtime_error("first shape of input and target must match.");
  }

  if (target.dtype() == kFloat64) {
    throw std::runtime_error("expects class IDs with integral type");
  }
  
  int batch_size = target.shape(0);
  Tensor losses = empty({batch_size});
  for (size_t i = 0; i < batch_size; i++) {
    int id = target(i);
    losses(i) = -input(id);
  }
  
  // currerntly only implement the mean policy from pytorch
  return sum(losses) / batch_size;
}

}  // namespace abyss::nn