#ifndef ABYSS_FUNCTIONAL_H
#define ABYSS_FUNCTIONAL_H

#include <vector>

#include "abyss_export.h"
#include "types.h"
#include "tensor.h"

namespace abyss {
ABYSS_EXPORT Tensor empty(std::vector<int> shape, ScalarType dtype = kFloat64);
ABYSS_EXPORT Tensor full(std::vector<int> shape, Tensor fill_value, ScalarType dtype = kNone);

ABYSS_EXPORT Tensor arange(Tensor start, Tensor stop, Tensor step, ScalarType dtype = kNone);
ABYSS_EXPORT Tensor arange(Tensor stop, ScalarType dtype = kNone);

ABYSS_EXPORT Tensor concat(std::vector<Tensor> tensors, int axis = 0);
/**
 * nn building blocks
 */

ABYSS_EXPORT Tensor add(Tensor lhs, Tensor rhs);
ABYSS_EXPORT Tensor subtract(Tensor lhs, Tensor rhs);
ABYSS_EXPORT Tensor multiply(Tensor lhs, Tensor rhs);
ABYSS_EXPORT Tensor divide(Tensor lhs, Tensor rhs);

ABYSS_EXPORT Tensor matmul(Tensor lhs, Tensor rhs);

ABYSS_EXPORT Tensor sum(Tensor a, int axis = -1);
ABYSS_EXPORT Tensor exp(Tensor a);

/**
 * complex layer types
 * maybe move to layers
 */
// Tensor linear(const Tensor& x, const Tensor& A, const Tensor& b);
// Tensor conv2d(const Tensor& kernel, const Tensor& stride = 1);
}  // namespace abyss

#endif  // ABYSS_FUNCTIONS_H
