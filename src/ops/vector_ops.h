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
#include "core/dispatcher.h"
#include "core/iterator.h"
#include "core/traits.h"
#include "core/utility.h"
#include "core/visitor.h"
#include "tensor.h"

// #include "autograd/graph.h"
// #include "operation.h"
// #include "tensor.h"

namespace abyss::core {

/**
 * @brief Vector Visitor that acts as a parent class for all vector operations.
 *
 * This class has a protected `eval` method which calls the correct overloaded
 * function by specifying the function signature in the template parameter.
 */
class BinaryVectorVisitor
    : public VisitorBase,
      public Tensor,
      public BinaryVisitor<ArrayImpl<int32_t>, ArrayImpl<int32_t>>,
      public BinaryVisitor<ArrayImpl<int32_t>, ArrayImpl<double>>,
      public BinaryVisitor<ArrayImpl<double>, ArrayImpl<int32_t>>,
      public BinaryVisitor<ArrayImpl<double>, ArrayImpl<double>> {
 public:
  // static std::vector<int> calc_output_shape(std::vector<int> shape1,
  //                                           std::vector<int> shape2);

  BinaryVectorVisitor(ArrayDesc desc1, ArrayDesc desc2)
      : desc1_{desc1}, desc2_{desc2} {}

 protected:
  // template <
  //     typename T1, typename T2, typename OutTp = std::common_type_t<T1, T2>,
  //     typename Callable = void(const T1*, const T2*, const size_t&, OutTp*)>
  // void eval(ArrayImpl<T1>* a, ArrayImpl<T2>* b, Callable fn) {
  //   desc_.shape = calc_output_shape(desc1_.shape, desc2_.shape);
  //   desc_.strides = shape2strides(desc_.shape);

  //   size_t output_size = shape2size(desc_.shape);
  //   auto out = std::make_shared<ArrayImpl<OutTp>>(output_size);

  //   ArrayImpl<T1> a_matched(output_size);
  //   ArrayImpl<T2> b_matched(output_size);
  //   broadcast_copy(a->begin(), a->end(), desc1_, a_matched.begin(), desc_);
  //   broadcast_copy(b->begin(), b->end(), desc2_, b_matched.begin(), desc_);

  //   fn(a_matched.data(), b_matched.data(), output_size, out->data());

  //   dtype_ = stypeof<OutTp>();
  //   data_ = out;
  // }

  /**
   * @brief new eval function that incorporates broadcast to the backend.
   *
   * This is done by sending index arrays as extra parameters in-order to
   * traverse the original data without copying.
   * Making a large function is not ideal but neccessary for computing
   * the output shape and strides as well as the broadcast arrangements.
   */
  template <typename T1, typename T2,
            typename OutTp = std::common_type_t<T1, T2>,
            typename Callable = void(const T1*, const size_t*, const T2*,
                                     const size_t*, const size_t, OutTp*)>
  void broadcast_eval(ArrayImpl<T1>* a, ArrayImpl<T2>* b, Callable fn) {
    // 1. sort out the output array description and broadcasting dimensions
    resolve_broadacast();
    // 2. compute the indices for the input arrays
    // generate indices for the backend
    size_t output_size = shape2size(desc_.shape);

    // bundle the properties together so we can iterate over them
    std::array<size_t, 2> offsets = {desc1_.offset, desc2_.offset};
    std::array<std::vector<int>, 2> shapes = {desc1_.shape, desc2_.shape};
    std::array<std::vector<int>, 2> strides = {desc1_.strides, desc2_.strides};

    std::vector<std::vector<size_t>> ids = {
        std::vector<size_t>(output_size, 0),
        std::vector<size_t>(output_size, 0)};
    // std::vector<std::vector<int>> coords = {std::vector<int>(max_dim, 0),
    // std::vector<int>(max_dim, 0)};
    std::vector<int> coords(desc_.shape.size(), 0);

    for (size_t i = 0; i < output_size; i++) {
      // loop through lhs and rhs
      coords = unravel_index(i, desc_.shape);
      for (size_t xpr = 0; xpr < 2; xpr++) {
        size_t offset = offsets[xpr];
        for (size_t j = 0; j < coords.size(); j++) {
          offset += coords[j] * strides[xpr][j];
        }

        ids[xpr][i] = offset;  // store offset
      }
    }

    // 3. call the backend function and get the result
    auto out = std::make_shared<ArrayImpl<OutTp>>(output_size);

    fn(a->data(), ids[0].data(), b->data(), ids[1].data(), output_size,
       out->data());

    dtype_ = stypeof<OutTp>();
    data_ = out;
  }

 private:
  ArrayDesc desc1_;
  ArrayDesc desc2_;

  /**
   * @brief update input descriptions to match broadcast.
   *
   * This function modifies the input descriptions (`desc1_`, `desc2_`)
   * and the output description `desc_`
   */
  void resolve_broadacast() {
    size_t dim1 = desc1_.shape.size();
    size_t dim2 = desc2_.shape.size();
    size_t max_dim = std::max(dim1, dim2);
    desc_.offset = 0;
    desc_.shape.assign(max_dim, 1);
    desc_.strides.assign(max_dim, 0);

    // pad to the same dimensions
    desc1_.shape.insert(desc1_.shape.begin(), max_dim - dim1, 1);
    desc1_.strides.insert(desc1_.strides.begin(), max_dim - dim1, 0);
    desc2_.shape.insert(desc2_.shape.begin(), max_dim - dim2, 1);
    desc2_.strides.insert(desc2_.strides.begin(), max_dim - dim2, 0);

    for (int d = max_dim - 1; d >= 0; d--) {
      // update output shape to the common size
      desc_.shape[d] = std::max(desc1_.shape[d], desc2_.shape[d]);

      int broadcasted_size =
          desc_.shape[d] / std::min(desc1_.shape[d], desc2_.shape[d]);
      if (broadcasted_size != 1) {
        // determine who needs broadcasting and set stride for the dimension
        // correctly.
        if (desc1_.shape[d] == 1) {
          desc1_.strides[d] = 0;
        } else if (desc2_.shape[d] == 1) {
          desc2_.strides[d] = 0;
        } else {
          throw std::runtime_error("BinaryVectorVisitor: not broadcastable");
        }
      }
    }

    desc_.strides = shape2strides(desc_.shape);
  }
};

class UnaryVectorVisitor : public VisitorBase,
                           public Tensor,
                           public UnaryVisitor<ArrayImpl<uint8_t>>,
                           public UnaryVisitor<ArrayImpl<int32_t>>,
                           public UnaryVisitor<ArrayImpl<double>> {
 public:
  UnaryVectorVisitor(ArrayDesc desc) : in_desc_{desc} {}

 protected:
  ArrayDesc in_desc_;

  template <typename T, typename Callable = void(const T*, const size_t&, T*)>
  void eval(ArrayImpl<T>* arr, Callable fn) {
    size_t output_size = shape2size(in_desc_.shape);
    auto out = std::make_shared<ArrayImpl<T>>(output_size);

    std::vector<int> coords(in_desc_.shape.size(), 0);
    std::vector<int> ids(output_size, 0);
    for (size_t i = 0; i < output_size; i++) {
      coords = unravel_index(i, in_desc_.shape);
      size_t offset = in_desc_.offset;
      for (size_t j = 0; j < coords.size(); j++) {
        offset += coords[j] * in_desc_.strides[j];
      }

      ids[i] = offset;
    }

    fn(arr.data(), ids, output_size, out->data());

    dtype_ = stypeof<T>();
    desc_ = in_desc_;
    data_ = out;
  }
};

/**
 * Arithmetic functions
 */

// class AddVisitor : public BinaryVectorVisitor {
class AddVisitor : public BinaryVectorVisitor {
 public:
  AddVisitor(ArrayDesc desc1, ArrayDesc desc2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

class SubtractVisitor : public BinaryVectorVisitor {
 public:
  SubtractVisitor(ArrayDesc desc1, ArrayDesc desc2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

class MultiplyVisitor : public BinaryVectorVisitor {
 public:
  MultiplyVisitor(ArrayDesc desc1, ArrayDesc desc2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

class DivideVisitor : public BinaryVectorVisitor {
 public:
  DivideVisitor(ArrayDesc desc1, ArrayDesc desc2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

/**
 * Comparison
 */
class EqualVisitor : public BinaryVectorVisitor {
 public:
  EqualVisitor(ArrayDesc desc1, ArrayDesc desc2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

class NotEqualVisitor : public BinaryVectorVisitor {
 public:
  NotEqualVisitor(ArrayDesc desc1, ArrayDesc desc2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;
};

}  // namespace abyss::core
#endif