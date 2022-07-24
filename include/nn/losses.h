#ifndef ABYSS_NN_LOSSES_H
#define ABYSS_NN_LOSSES_H

#include "module.h"

namespace abyss::nn {
// class ABYSS_EXPORT CrossEntropyLoss : public Module {
//   public:
//   CrossEntropyLoss() = default;
  
//   Tensor forward(Tensor input, Tensor target) override;
// };
class ABYSS_EXPORT NLLLoss : public Module {
  public:
  NLLLoss() = default;
  
  Tensor forward(Tensor input, Tensor target) override;
};
}

#endif