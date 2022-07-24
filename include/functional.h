#ifndef ABYSS_FUNCTIONAL_H
#define ABYSS_FUNCTIONAL_H

#include <tuple>
#include <vector>

#include "abyss_export.h"
// #include "types.h"
#include "scalartype.h"
// #include "core/traits.h"
#include "core/dispatcher.h"
#include "core/visitor.h"
// #include "ops/dtype_ops.h"
#include "tensor.h"

namespace abyss {
ABYSS_EXPORT Tensor empty(std::vector<int> shape, ScalarType dtype = kFloat64);
ABYSS_EXPORT Tensor full(std::vector<int> shape, Tensor fill_value,
                         ScalarType dtype = kNone);

template <typename T1, typename T2, typename T3>
ABYSS_EXPORT Tensor arange(T1 start, T2 stop, T3 step, ScalarType dtype = kNone) {
  using namespace core;
  using common_t = std::common_type_t<T1, T2, T3>;

  /**
   * @brief New Arange Visitor
   * 
   * arange function will look messy if not implemented as a template (with 27 overloads).
   * The decision was made to move the visitor into the function so we don't have to include private headers
   * (namely: ops/dtype_ops.h)
   */
  struct ArangeImpl : public VisitorBase,
                      public Tensor,
                      public UnaryVisitor<DTypeImpl<uint8_t>>,
                      public UnaryVisitor<DTypeImpl<int32_t>>,
                      public UnaryVisitor<DTypeImpl<double>> {
    ArangeImpl(common_t start, common_t stop, common_t step) : start_{start}, stop_{stop}, step_{step} {}
    
    void visit(core::DTypeImpl<uint8_t>* dtype) override {
      auto arr = ArrayImpl<common_t>::from_range(start_, stop_, step_);
      desc_.shape = {static_cast<int>(arr.size())};
      desc_.strides = {1};
      data_ = std::make_shared<ArrayImpl<uint8_t>>(arr);
    }

    void visit(core::DTypeImpl<int32_t>* dtype) override {
      auto arr = ArrayImpl<common_t>::from_range(start_, stop_, step_);
      desc_.shape = {static_cast<int>(arr.size())};
      desc_.strides = {1};
      data_ = std::make_shared<ArrayImpl<int32_t>>(arr);
      // data_ = std::make_shared<ArrayImpl<int32_t>();
    }
    void visit(core::DTypeImpl<double>* dtype) override {
      auto arr = ArrayImpl<common_t>::from_range(start_, stop_, step_);
      desc_.shape = {static_cast<int>(arr.size())};
      desc_.strides = {1};
      data_ = std::make_shared<ArrayImpl<double>>(arr);
      // data_ = std::make_shared<ArrayImpl<int32_t>();
    }

    private:

    common_t start_;
    common_t stop_;
    common_t step_;
  };

  ArangeImpl range_vis(start, stop, step);
  TypeDispatcher<ScalarType> dispatcher(dtype);

  dispatcher.accept(&range_vis);

  return range_vis;
}

template <typename T>
ABYSS_EXPORT Tensor arange(T stop, ScalarType dtype = kNone) {
  return arange(0, stop, 1, dtype);
}

ABYSS_EXPORT Tensor randn(std::vector<int> shape, ScalarType dtype = kNone);

ABYSS_EXPORT Tensor concat(std::vector<Tensor> tensors, int axis = 0);
/**
 * nn building blocks
 */

ABYSS_EXPORT Tensor add(Tensor lhs, Tensor rhs);
ABYSS_EXPORT Tensor subtract(Tensor lhs, Tensor rhs);
ABYSS_EXPORT Tensor multiply(Tensor lhs, Tensor rhs);
ABYSS_EXPORT Tensor divide(Tensor lhs, Tensor rhs);

ABYSS_EXPORT Tensor matmul(Tensor lhs, Tensor rhs);

ABYSS_EXPORT Tensor exp(Tensor a);
ABYSS_EXPORT Tensor log(Tensor a);

ABYSS_EXPORT Tensor sum(Tensor a /*, axis = None*/);
ABYSS_EXPORT Tensor sum(Tensor a, int axis);

ABYSS_EXPORT Tensor negative(Tensor a);

/**
 * complex layer types
 * maybe move to layers
 */
// Tensor linear(const Tensor& x, const Tensor& A, const Tensor& b);
// Tensor conv2d(const Tensor& kernel, const Tensor& stride = 1);
}  // namespace abyss

#endif  // ABYSS_FUNCTIONS_H
