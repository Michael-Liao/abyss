#ifndef ABYSS_CORE_MERGE_OPS_H
#define ABYSS_CORE_MERGE_OPS_H

#include <limits>
#include <memory>
#include <vector>

#include "abyss_export.h"
#include "backend/reduction.h"
#include "core/array.h"
#include "core/traits.h"
#include "core/utility.h"
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

    desc_.shape = calc_output_shape(shape1_, shape2_, axis_);

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
    desc_.strides = shape2strides(desc_.shape);
    data_ = out;
  }
};

class ReductionVisitor : public VisitorBase,
                         public Tensor,
                         public UnaryVisitor<ArrayImpl<bool>>,
                         public UnaryVisitor<ArrayImpl<uint8_t>>,
                         public UnaryVisitor<ArrayImpl<int32_t>>,
                         public UnaryVisitor<ArrayImpl<double>> {
 public:
  static const int kNoAxis = std::numeric_limits<int>::max();

  ReductionVisitor(ArrayDesc desc, int axis = kNoAxis);

 protected:
  int axis_;
  ArrayDesc in_desc_;
  // static const int kNoAxis = std::numeric_limits<int>::min();

  template <typename T, typename Callable = void(const T*, int, size_t, T*)>
  void eval(ArrayImpl<T>* a, Callable fn) {
    // generate output description from the input description and axis
    gen_output_desc();

    size_t output_size = shape2size(desc_.shape);
    auto arr = std::make_shared<ArrayImpl<T>>(output_size);
    arr->zero();

    int size = 0;
    int stride = 0;
    if (axis_ == kNoAxis) {
      size = shape2size(in_desc_.shape);
      stride = 1;

      fn(a->data(), stride, size, arr->data());
    } else {
      size = in_desc_.shape[axis_];
      stride = in_desc_.strides[axis_];
      // modify shape so the reduced axis has shape of 1
      // this ensures the coords have the correct dimensions
      in_desc_.shape[axis_] = 1;
      // calculate
      for (size_t i = 0; i < output_size; i++) {
        auto coords = unravel_index(i, in_desc_.shape);
        size_t offset = in_desc_.offset;
        // for (size_t j = 0; j < coords.size(); j++) {
        //   offset += coords[j] * in_desc_.strides[j];
        // }
        for (size_t j = 0; j < coords.size(); j++) {
          offset += coords[j] * in_desc_.strides[j];
        }

        fn(a->data() + offset, stride, size, arr->data() + i);
      }
    }

    dtype_ = stypeof<T>();
    data_ = arr;
  }

 private:
  // int axis_;
  // ArrayDesc in_desc_;

  void gen_output_desc() {
    if (axis_ == kNoAxis) {
      desc_.shape = {1};
    } else {
      for (size_t i = 0; i < in_desc_.shape.size(); i++) {
        // if (i == axis_) {
        //   desc_.shape[i] = 1;
        // } else {
        //   desc_.shape[i] = in_desc_.shape[i];
        // }
        
        if (axis_ != i) {
          desc_.shape.emplace_back(in_desc_.shape[i]);
        }
      }
    }

    desc_.strides = shape2strides(desc_.shape);
  }
};

class AllVisitor final : public VisitorBase,
                         public Tensor,
                         public UnaryVisitor<ArrayImpl<bool>>,
                         public UnaryVisitor<ArrayImpl<uint8_t>>,
                         public UnaryVisitor<ArrayImpl<int32_t>>,
                         public UnaryVisitor<ArrayImpl<double>> {
 public:
  // static const int kNoAxis = std::numeric_limits<int>::max();
  static std::vector<int> calc_output_shape(std::vector<int> shape, int axis);

  AllVisitor(ArrayDesc desc, int axis = kNoAxis);

  void visit(ArrayImpl<bool>*) override;
  void visit(ArrayImpl<uint8_t>*) override;
  void visit(ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*) override;

 private:
  static const int kNoAxis = std::numeric_limits<int>::max();
  int axis_;
  ArrayDesc in_desc_;

  template <typename T>
  void eval(ArrayImpl<T>* arr) {
    // output shape cal
    desc_.shape = calc_output_shape(in_desc_.shape, axis_);
    size_t in_size = shape2size(in_desc_.shape);

    size_t output_size = shape2size(desc_.shape);
    auto out = std::make_shared<ArrayImpl<T>>(output_size);

    size_t stride =
        (axis_ == kNoAxis) ? 1 : shape2strides(in_desc_.shape)[axis_];
    for (size_t i = 0; i < output_size; i++) {
      // safe boolean conversion (for all arithmetics types)
      out->at(i) =
          std::all_of(arr->begin(), arr->end(), [](T a) { return (a != 0); });

      // out->at(i);
      // auto coords = unravel_index(i, in_desc_.shape);
      // size_t offset = in_desc_.offset;
      // for (size_t i = 0; i < in_size; i++) {
      //   offset +=
      // }
    }

    dtype_ = stypeof<T>();
    desc_.strides = shape2strides(desc_.shape);
    data_ = out;
  }
};

class SumVisitor : public ReductionVisitor {
 public:
  SumVisitor(ArrayDesc desc);
  SumVisitor(ArrayDesc desc, int axis);

  void visit(ArrayImpl<bool>*) override;
  void visit(ArrayImpl<uint8_t>*) override;
  void visit(ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*) override;
};

}  // namespace abyss::core

#endif