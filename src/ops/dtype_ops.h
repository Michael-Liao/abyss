#ifndef ABYSS_CORE_DTYPE_OPS_H
#define ABYSS_CORE_DTYPE_OPS_H

#include <complex>
#include <memory>
#include <random>
#include <vector>

#include "abyss_export.h"
#include "core/array.h"
#include "core/dtype.h"
#include "core/traits.h"
#include "core/utility.h"
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
    desc_.shape = shape_;
    desc_.strides = shape2strides(shape_);

    size_t output_size = shape2size(shape_);
    data_ = std::make_shared<ArrayImpl<T>>(output_size);
    flags_[core::FlagId::kIsContiguous] = true;
    flags_[core::FlagId::kOwnsData] = true;
    flags_[core::FlagId::kIsLeaf] = true;
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
    desc_.offset = 0;
    desc_.shape = shape_;
    desc_.strides = shape2strides(shape_);
    data_ = std::make_shared<ArrayImpl<T2>>(output_size, value->at(0));
    flags_[core::FlagId::kIsContiguous] = true;
    flags_[core::FlagId::kOwnsData] = true;
    flags_[core::FlagId::kIsLeaf] = true;
  }
};

// template <typename T, std::enable_if_t<is_supported_dtype<T>::value, bool> =
// true> template <typename T> class ArangeVisitor final : public VisitorBase,
//                             public Tensor,
//                             public UnaryVisitor<DTypeImpl<bool>>,
//                             public UnaryVisitor<DTypeImpl<uint8_t>>,
//                             public UnaryVisitor<DTypeImpl<int32_t>>,
//                             public UnaryVisitor<DTypeImpl<double>> {
//  public:
//   static_assert(is_supported_dtype<T>::value, "ArangeVisitor must be a
//   supported type");
//   // ArangeVisitor() = default;
//   ArangeVisitor(T start, T stop, T step)
//       : start_{start}, stop_{stop}, step_{step} {}

//   void visit(DTypeImpl<bool>* dtype) override { eval(dtype); }
//   void visit(DTypeImpl<uint8_t>* dtype) override { eval(dtype); }
//   void visit(DTypeImpl<int32_t>* dtype) override { eval(dtype); }
//   void visit(DTypeImpl<double>* dtype) override { eval(dtype); }

//  private:
//   T start_ = 0;
//   T stop_ = 1;
//   T step_ = 1;

//   template <typename TgtTp>
//   void eval(DTypeImpl<TgtTp>* dtype) {
//     auto out = ArrayImpl<T>::from_range(start_, stop_, step_);

//     dtype_ = dtype;
//     desc_.shape = {static_cast<int>(out.size())};
//     desc_.strides = {1};
//     data_ = std::make_shared<ArrayImpl<TgtTp>>(out);
//   }
// };
class RandNormalVisitor : public VisitorBase,
                          public Tensor,
                          public UnaryVisitor<DTypeImpl<double>> {
 public:
  RandNormalVisitor(std::vector<int> shape);
  void visit(DTypeImpl<double>* dtype) override;

 private:
  std::vector<int> shape_;

  template <typename TgtTp>
  void eval(DTypeImpl<TgtTp>* dtype) {
    dtype_ = dtype;
    desc_.offset = 0;
    desc_.shape = shape_;
    desc_.strides = shape2strides(shape_);

    size_t output_size = shape2size(shape_);
    auto arr = std::make_shared<ArrayImpl<TgtTp>>(output_size);

    std::random_device rd;
    std::mt19937 rng(rd());
    // rng.seed(0); // currently don't know how to set seed by user
    std::normal_distribution<TgtTp> dist(0, 1);

    for (size_t i = 0; i < output_size; i++) {
      arr->at(i) = dist(rng);
    }
    
    data_ = arr;

    flags_[core::FlagId::kIsContiguous] = true;
    flags_[core::FlagId::kOwnsData] = true;
    flags_[core::FlagId::kIsLeaf] = true;
  }
};

}  // namespace abyss::core
#endif