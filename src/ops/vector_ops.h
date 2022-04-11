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

  VectorVisitor(std::vector<int> shape1, std::vector<int> shape2)
      : shape1_{shape1}, shape2_{shape2} {}

 protected:
  template <
      typename T1, typename T2, typename OutTp = std::common_type_t<T1, T2>,
      typename Callable = void(const T1*, const T2*, const size_t&, OutTp*)>
  void eval(ArrayImpl<T1>* a, ArrayImpl<T2>* b, Callable fn) {
    shape_ = calc_output_shape(shape1_, shape2_);

    size_t output_size = shape2size(shape_);
    auto out = std::make_shared<ArrayImpl<OutTp>>(output_size);

    ArrayImpl<T1> a_matched(output_size);
    ArrayImpl<T2> b_matched(output_size);
    broadcast_copy(a->begin(), a->end(), shape1_, a_matched.begin(), shape_);
    broadcast_copy(b->begin(), b->end(), shape2_, b_matched.begin(), shape_);

    fn(a_matched.data(), b_matched.data(), output_size, out->data());

    dtype_ = stypeof<OutTp>();
    strides_ = shape2strides(shape_);
    data_ = out;
  }

 private:
  std::vector<int> shape1_;
  std::vector<int> shape2_;
};

/**
 * Arithmetic functions
 */

// class AddVisitor : public VectorVisitor {
class AddVisitor : public VectorVisitor {
 public:
  AddVisitor(std::vector<int> shape1, std::vector<int> shape2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

class SubtractVisitor : public VectorVisitor {
 public:
  SubtractVisitor(std::vector<int> shape1, std::vector<int> shape2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

class MultiplyVisitor : public VectorVisitor {
 public:
  MultiplyVisitor(std::vector<int> shape1, std::vector<int> shape2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

class DivideVisitor : public VectorVisitor {
 public:
  DivideVisitor(std::vector<int> shape1, std::vector<int> shape2);

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
  EqualVisitor(std::vector<int> shape1, std::vector<int> shape2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

class NotEqualVisitor : public VectorVisitor {
 public:
  NotEqualVisitor(std::vector<int> shape1, std::vector<int> shape2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

}  // namespace abyss::core
#endif