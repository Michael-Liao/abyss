#ifndef ABYSS_CORE_MERGE_OPS_H
#define ABYSS_CORE_MERGE_OPS_H

#include <limits>
#include <memory>
#include <vector>

#include "abyss_export.h"
#include "core/array.h"
#include "core/traits.h"
#include "core/visitor.h"
#include "tensor.h"

namespace abyss::core {
// std::vector<int> concat_calc_output_shape(std::vector<int> a,
//                                           std::vector<int> b,
//                                           int ignore_axis = 0);
class ConcatVisitor final
    : public VisitorBase,
      public Tensor,
      public BinaryVisitor<ArrayImpl<int32_t>, ArrayImpl<int32_t>>,
      public BinaryVisitor<ArrayImpl<int32_t>, ArrayImpl<double>>,
      public BinaryVisitor<ArrayImpl<double>, ArrayImpl<int32_t>>,
      public BinaryVisitor<ArrayImpl<double>, ArrayImpl<double>> {
 public:
  static std::vector<int> calc_output_shape(std::vector<int> a,
                                            std::vector<int> b,
                                            int ignore_axis = 0);

  ConcatVisitor(std::vector<int> shape1, std::vector<int> shape2, int axis = 0);
  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;

 private:
  // int element_size_;
  int axis_ = 0;
  std::vector<int> shape1_;
  std::vector<int> shape2_;

  // bool check_shape(const std::vector<int>& a, const std::vector<int>& b,
  //                  int ignore_axis = 0);
  // std::vector<int> calc_output_shape(const std::vector<int>& a,
  //                                    const std::vector<int>& b,
  //                                    int ignore_axis = 0);

  template <typename T1, typename T2,
            typename OutTp = std::common_type_t<T1, T2>>
  void eval(ArrayImpl<T1>* a, ArrayImpl<T2>* b) {
    auto out = std::make_shared<ArrayImpl<OutTp>>(a->size() + b->size());

    shape_ = calc_output_shape(shape1_, shape2_, axis_);

    const int gap_b = std::accumulate(shape2_.rbegin(), shape2_.rend() - axis_,
                                      1, std::multiplies<int>());
    const int gap_a = std::accumulate(shape1_.rbegin(), shape1_.rend() - axis_,
                                      1, std::multiplies<int>());

    // fill a
    int i = 0;
    while (i < a->size()) {
      out->at(i + i / gap_a * gap_b) = a->at(i);

      i++;
    }

    // fill b
    i = 0;
    while (i < b->size()) {
      out->at(i + (i / gap_b + 1) * gap_a) = b->at(i);

      i++;
    }

    dtype_ = stypeof<int32_t>();
    // output_shape_ = calc_output_shape(shape1_, shape2_, axis_);
    strides_ = shape2strides(shape_);
    data_ = out;
  }
};

class AllVisitor final : public VisitorBase,
                         public Tensor,
                         public UnaryVisitor<ArrayImpl<bool>>,
                         public UnaryVisitor<ArrayImpl<uint8_t>>,
                         public UnaryVisitor<ArrayImpl<int32_t>>,
                         public UnaryVisitor<ArrayImpl<double>> {
 public:
  static const int kNoAxis = std::numeric_limits<int>::max();
  static std::vector<int> calc_output_shape(std::vector<int> shape, int axis);

  AllVisitor(std::vector<int> shape, int axis = kNoAxis);

  void visit(ArrayImpl<bool>*) override;
  void visit(ArrayImpl<uint8_t>*) override;
  void visit(ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*) override;

 private:
  int axis_;
  std::vector<int> in_shape_;

  template <typename T>
  void eval(ArrayImpl<T>* arr) {
    // output shape cal
    shape_ = calc_output_shape(in_shape_, axis_);

    size_t output_size = shape2size(shape_);
    auto out = std::make_shared<ArrayImpl<T>>(output_size);

    size_t stride = (axis_ == kNoAxis) ? 1 : shape2strides(in_shape_)[axis_];
    for (size_t i = 0; i < output_size; i++) {
      // safe boolean conversion (for all arithmetics types)
      out->at(i) =
          std::all_of(arr->begin(), arr->end(), [](T a) { return (a != 0); });
    }

    dtype_ = stypeof<T>();
    strides_ = shape2strides(shape_);
    data_ = out;
  }
};

}  // namespace abyss::core

#endif