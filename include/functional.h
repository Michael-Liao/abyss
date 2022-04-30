#ifndef ABYSS_FUNCTIONAL_H
#define ABYSS_FUNCTIONAL_H

#include <vector>

#include "abyss_export.h"
// #include "types.h"
#include "scalartype.h"
// #include "core/traits.h"
#include "tensor.h"
#include "ops/dispatcher.h"
#include "ops/dtype_ops.h"

namespace abyss {
ABYSS_EXPORT Tensor empty(std::vector<int> shape, ScalarType dtype = kFloat64);
ABYSS_EXPORT Tensor full(std::vector<int> shape, Tensor fill_value, ScalarType dtype = kNone);

template <typename T1, typename T2, typename T3>
ABYSS_EXPORT Tensor arange(T1 start, T2 stop, T3 step, ScalarType dtype = kNone) {
  using common_t = std::common_type_t<T1, T2, T3>;

  if (dtype == kNone) {
    dtype = stypeof<common_t>();
  }

  core::Dispatcher<ScalarType> dtype_dispatch(dtype);

  core::ArangeVisitor<common_t> arange_visitor{start, stop, step};
  dtype_dispatch.accept(&arange_visitor);
  
  return arange_visitor;
}

template <typename T>
ABYSS_EXPORT Tensor arange(T stop, ScalarType dtype = kNone) {
  return arange(0, stop, 1, dtype);
}

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
