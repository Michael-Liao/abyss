#include "nn/activation.h"

#include "functional.h"
#include "operators.h"

namespace abyss::nn {

LogSoftmax::LogSoftmax(int axis) : axis_{axis} {}

Tensor LogSoftmax::forward(Tensor input) {
  return log(exp(input) / sum(exp(input), axis_));
}

}