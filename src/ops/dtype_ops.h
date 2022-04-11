#ifndef ABYSS_CORE_DTYPE_OPS_H
#define ABYSS_CORE_DTYPE_OPS_H

#include <complex>
#include <memory>
#include <vector>

#include "abyss_export.h"
#include "core/array.h"
#include "core/dtype.h"
#include "core/visitor.h"
#include "tensor.h"
#include "types.h"

namespace abyss::core {

// class ABYSS_EXPORT EmptyVisitor : public VisitorBase,
class EmptyVisitor final : public VisitorBase,
                           public Tensor,
                           public UnaryVisitor<DTypeImpl<bool>>,
                           public UnaryVisitor<DTypeImpl<int32_t>>,
                           public UnaryVisitor<DTypeImpl<double>> {
 public:
  EmptyVisitor(std::vector<int> shape);
  ~EmptyVisitor() = default;

  void visit(DTypeImpl<bool>*) override;
  void visit(DTypeImpl<int32_t>*) override;
  void visit(DTypeImpl<double>*) override;

 private:
  std::vector<int> shape_;

  // template <typename T>
  // void eval(DTypeImpl<T>* dtype) {
  //   result_dtype_ = dtype;
  //   output_shape_ = shape_;
  //   output_strides_ = shape2strides(shape_);

  //   size_t output_size = shape2size(shape_);
  //   shared_data_ = std::make_shared<ArrayImpl<T>>(output_size);
  // }

  template <typename T>
  void eval(DTypeImpl<T>* dtype) {
    dtype_ = dtype;
    Tensor::shape_ = shape_;
    strides_ = shape2strides(shape_);

    size_t output_size = shape2size(shape_);
    data_ = std::make_shared<ArrayImpl<T>>(output_size);
  }
};

class FullVisitor final
    : public VisitorBase,
      public Tensor,
      public BinaryVisitor<ArrayImpl<bool>, DTypeImpl<bool>>,
      public BinaryVisitor<ArrayImpl<int32_t>, DTypeImpl<bool>>,
      public BinaryVisitor<ArrayImpl<int32_t>, DTypeImpl<int32_t>>,
      public BinaryVisitor<ArrayImpl<double>, DTypeImpl<int32_t>>,
      public BinaryVisitor<ArrayImpl<int32_t>, DTypeImpl<double>>,
      public BinaryVisitor<ArrayImpl<double>, DTypeImpl<double>> {
 public:
  FullVisitor(std::vector<int> shape);
  ~FullVisitor() = default;

  void visit(ArrayImpl<bool>*, DTypeImpl<bool>*) override;
  void visit(ArrayImpl<int32_t>*, DTypeImpl<bool>*) override;
  void visit(ArrayImpl<int32_t>*, DTypeImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, DTypeImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, DTypeImpl<double>*) override;
  void visit(ArrayImpl<double>*, DTypeImpl<double>*) override;

 private:
  std::vector<int> shape_;
  template <typename T1, typename T2>
  void eval(ArrayImpl<T1>* value, DTypeImpl<T2>* dtype) {
    if (value->size() > 1)
      throw std::runtime_error("fill value should be a scalar.");

    size_t output_size = shape2size(shape_);

    dtype_ = dtype;
    Tensor::shape_ = shape_;
    strides_ = shape2strides(shape_);
    data_ = std::make_shared<ArrayImpl<T2>>(output_size, value->at(0));
  }
};

class ArangeVisitor final
    : public VisitorBase,
      public Tensor,
      public BinaryVisitor<ArrayImpl<int32_t>, DTypeImpl<int32_t>>,
      public BinaryVisitor<ArrayImpl<int32_t>, DTypeImpl<double>>,
      public BinaryVisitor<ArrayImpl<double>, DTypeImpl<int32_t>>,
      public BinaryVisitor<ArrayImpl<double>, DTypeImpl<double>> {
 public:
  ArangeVisitor() = default;

  void visit(ArrayImpl<int32_t>*, DTypeImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, DTypeImpl<double>*) override;
  void visit(ArrayImpl<double>*, DTypeImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, DTypeImpl<double>*) override;

 private:
  template <typename T1, typename T2>
  void eval(ArrayImpl<T1>* arr, DTypeImpl<T2>* dtype) {
    auto out = ArrayImpl<T2>::from_range(arr->at(0), arr->at(1), arr->at(2));

    dtype_ = dtype;
    Tensor::shape_ = {static_cast<int>(out.size())};
    strides_ = {1};
    data_ = std::make_shared<ArrayImpl<T2>>(out);
  }
};

}  // namespace abyss::core
#endif