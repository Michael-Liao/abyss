#ifndef ABYSS_CORE_MATRIX_OPS_H
#define ABYSS_CORE_MATRIX_OPS_H

#include <tuple>
#include <vector>

#include "backend/matmul.h"
#include "core/array.h"
#include "core/traits.h"
#include "core/utility.h"
#include "core/visitor.h"
#include "tensor.h"

namespace abyss::core {

/**
 * @brief Matmul computation
 * 
 * This currently only resolves broadcast but not view schematics
 */
class MatmulVisitor final
    : public VisitorBase,
      public Tensor,
      public BinaryVisitor<ArrayImpl<int32_t>, ArrayImpl<int32_t>>,
      public BinaryVisitor<ArrayImpl<int32_t>, ArrayImpl<double>>,
      public BinaryVisitor<ArrayImpl<double>, ArrayImpl<int32_t>>,
      public BinaryVisitor<ArrayImpl<double>, ArrayImpl<double>> {
 public:
  /**
   * @brief matmul output shape calculation
   * @return a tuple of (output_shape, broadcast_shape1, broadcast_shape2)
   */
  std::tuple<std::vector<int>, std::vector<int>,
             std::vector<int>> static calc_output_shape(std::vector<int> shape1,
                                                        std::vector<int>
                                                            shape2);
  MatmulVisitor(ArrayDesc desc1, ArrayDesc desc2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;

  template <typename T1, typename T2>
  void eval(ArrayImpl<T1>* a, ArrayImpl<T2>* b) {
    using result_t = std::common_type_t<T1, T2>;
    // std::vector<int> bc_shp1;
    // std::vector<int> bc_shp2;
    ArrayDesc bc_desc1;
    ArrayDesc bc_desc2;
    std::tie(desc_.shape, bc_desc1.shape, bc_desc2.shape) =
        calc_output_shape(desc1_.shape, desc2_.shape);

    bc_desc1.strides = shape2strides(bc_desc1.shape);
    bc_desc2.strides = shape2strides(bc_desc2.shape);

    // broadcast as needed
    ArrayImpl<T1> a_matched(shape2size(bc_desc1.shape));
    broadcast_copy(a->begin(), a->end(), desc1_, a_matched.begin(), bc_desc1);
    ArrayImpl<T2> b_matched(shape2size(bc_desc1.shape));
    broadcast_copy(b->begin(), b->end(), desc2_, b_matched.begin(), bc_desc2);

    size_t output_size = shape2size(desc_.shape);
    auto c = std::make_shared<ArrayImpl<result_t>>(output_size);

    // int n_stacks = std::accumulate(desc_.shape.rbegin() + 2,
    // desc_.shape.rend(),
    //                                1, std::multiplies<>());

    int n_stacks = 1;
    if (desc_.shape.size() >= 2) {
      n_stacks = std::accumulate(desc_.shape.rbegin() + 2, desc_.shape.rend(),
                                 1, std::multiplies<>());
    }
    // const int rows = *(bc_desc1.shape.rbegin() + 1);
    int rows = 1;
    if (bc_desc1.shape.size() > 1) {
      rows = *(bc_desc1.shape.rbegin() + 1);
    }
    const int common = *bc_desc1.shape.rbegin();
    int cols = 1;
    if (bc_desc1.shape.size() > 1) {
      cols = *bc_desc2.shape.rbegin();
    }

    auto data_it1 = a_matched.begin(rows * common);
    auto data_it2 = b_matched.begin(common * cols);
    auto data_oit = c->begin(rows * cols);
    for (int i = 0; i < n_stacks; i++) {
      // actual calculation
      backend::matmul(&(*data_it1), &(*data_it2), rows, common, cols,
                      &(*data_oit));

      // advance
      data_it1++;
      data_it2++;
      data_oit++;
    }

    dtype_ = stypeof<result_t>();
    desc_.strides = shape2strides(desc_.shape);
    data_ = c;
  }

 private:
  ArrayDesc desc1_;
  ArrayDesc desc2_;

  /**
   * @brief function to resolve broadcast
   */
  void resolve_broadcast() {
    size_t dim1 = desc1_.shape.size();
    size_t dim2 = desc2_.shape.size();
    size_t max_dim = std::max(desc1_.shape.size(), desc2_.shape.size());

    desc_.offset = 0;
    desc_.shape.assign(max_dim, 1);

    if (max_dim == 1) {
      if (desc1_.shape != desc2_.shape) {
        throw std::runtime_error("MatMulVisitor: common shape mismatch [vec].");
      }

      desc_.shape.assign({1});
      desc_.strides.assign({1});

      return;
    }

    desc1_.shape.insert(desc1_.shape.begin(), max_dim - dim1, 1);
    desc1_.strides.insert(desc1_.strides.begin(), max_dim - dim1, 0);

    desc2_.shape.insert(desc2_.shape.end(), max_dim - dim2, 1);
    desc2_.strides.insert(desc2_.strides.end(), max_dim - dim2, 0);

    auto it1 = desc1_.shape.rbegin();
    auto it2 = desc2_.shape.rbegin();
    if (*it1 != *(it2 + 1)) {
      throw std::runtime_error("MatMulVisitor: common shape mismatch [mat].");
    }

    // assign the last 2 dimensions
    *(desc_.shape.rbegin() + 1) = *(it1 + 1);
    *desc_.shape.rbegin() = *it2;

    for (int d = max_dim - 1; d >= 2; d--) {
      desc_.shape[d] = std::max(desc1_.shape[d], desc2_.shape[d]);
      int broadcast_size =
          desc_.shape[d] / std::min(desc1_.shape[d], desc2_.shape[d]);
      if (broadcast_size != 1) {
        if (desc1_.shape[d] == 1) {
          desc1_.strides[d] = 0;
        } else if (desc2_.shape[d] == 1) {
          desc2_.strides[d] = 0;
        } else {
          throw std::runtime_error("MatMulVisitor: shape not broadcastable.");
        }
      }
    }

    desc_.strides = shape2strides(desc_.shape);
  }
};

}  // namespace abyss::core

#endif