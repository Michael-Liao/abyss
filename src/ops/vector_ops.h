#ifndef ABYSS_CORE_ARITH_OPS_H
#define ABYSS_CORE_ARITH_OPS_H

/**
 * @file arithmetics for Array
 * @details only use overloading instead of templates
 * so all backends will have the same interface
 */

#include <cstddef>
#include <memory>
// #include <tuple>
#include <vector>

// #include "abyss_export.h"
#include "backend/arithmetics.h"
#include "backend/comparison.h"
#include "core/array.h"
#include "core/utility.h"
#include "core/iterator.h"
#include "core/traits.h"
#include "core/visitor.h"
#include "tensor.h"

// #include "operation.h"
// #include "tensor.h"

namespace abyss::core {

/**
 * @brief Vector Visitor that acts as a parent class for all vector operations.
 *
 * This class has a protected `eval` method which calls the correct overloaded
 * function by specifying the function signature in the template parameter.
 */
class VectorVisitor
    : public VisitorBase,
      public Tensor,
      public BinaryVisitor<ArrayImpl<int32_t>, ArrayImpl<int32_t>>,
      public BinaryVisitor<ArrayImpl<int32_t>, ArrayImpl<double>>,
      public BinaryVisitor<ArrayImpl<double>, ArrayImpl<int32_t>>,
      public BinaryVisitor<ArrayImpl<double>, ArrayImpl<double>> {
 public:
  static std::vector<int> calc_output_shape(std::vector<int> shape1,
                                            std::vector<int> shape2);

  VectorVisitor(TensorDesc desc1, TensorDesc desc2)
      : desc1_{desc1}, desc2_{desc2} {}

 protected:
  template <
      typename T1, typename T2, typename OutTp = std::common_type_t<T1, T2>,
      typename Callable = void(const T1*, const T2*, const size_t&, OutTp*)>
  void eval(ArrayImpl<T1>* a, ArrayImpl<T2>* b, Callable fn) {
    desc_.shape = calc_output_shape(desc1_.shape, desc2_.shape);
    desc_.strides = shape2strides(desc_.shape);

    size_t output_size = shape2size(desc_.shape);
    auto out = std::make_shared<ArrayImpl<OutTp>>(output_size);

    ArrayImpl<T1> a_matched(output_size);
    ArrayImpl<T2> b_matched(output_size);
    broadcast_copy(a->begin(), a->end(), desc1_, a_matched.begin(), desc_);
    broadcast_copy(b->begin(), b->end(), desc2_, b_matched.begin(), desc_);

    fn(a_matched.data(), b_matched.data(), output_size, out->data());

    dtype_ = stypeof<OutTp>();
    data_ = out;
  }

 private:
  TensorDesc desc1_;
  TensorDesc desc2_;
  // std::vector<int> shape1_;
  // std::vector<int> shape2_;
};

/**
 * Arithmetic functions
 */

// class AddVisitor : public VectorVisitor {
class AddVisitor : public VectorVisitor {
 public:
  AddVisitor(TensorDesc desc1, TensorDesc desc2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

class SubtractVisitor : public VectorVisitor {
 public:
  SubtractVisitor(TensorDesc desc1, TensorDesc desc2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

class MultiplyVisitor : public VectorVisitor {
 public:
  MultiplyVisitor(TensorDesc desc1, TensorDesc desc2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

class DivideVisitor : public VectorVisitor {
 public:
  DivideVisitor(TensorDesc desc1, TensorDesc desc2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

/**
 * Comparison
 */
class EqualVisitor : public VectorVisitor {
 public:
  EqualVisitor(TensorDesc desc1, TensorDesc desc2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

class NotEqualVisitor : public VectorVisitor {
 public:
  NotEqualVisitor(TensorDesc desc1, TensorDesc desc2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

}  // namespace abyss::core
#endif