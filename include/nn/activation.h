#ifndef ABYSS_NN_ACTIVATION_H
#define ABYSS_NN_ACTIVATION_H

#include "abyss_export.h"
#include "module.h"

namespace abyss::nn {
class ABYSS_EXPORT LogSoftmax : public Module {
 public:
  LogSoftmax(int axis);

  Tensor forward(Tensor input) override;

 private:
  int axis_;
};
}  // namespace abyss::nn
#endif