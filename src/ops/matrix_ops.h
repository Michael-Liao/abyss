#ifndef ABYSS_CORE_MATRIX_OPS_H
#define ABYSS_CORE_MATRIX_OPS_H

#include <tuple>
#include <vector>

#include "core/array.h"
#include "core/utility.h"
#include "core/visitor.h"
#include "backend/matmul.h"
#include "core/traits.h"
#include "tensor.h"

namespace abyss::core {

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
  MatmulVisitor(TensorDesc desc1, TensorDesc desc2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;

  template <typename T1, typename T2>
  void eval(ArrayImpl<T1>* a, ArrayImpl<T2>* b) {
    using result_t = std::common_type_t<T1, T2>;
    // std::vector<int> bc_shp1;
    // std::vector<int> bc_shp2;
    TensorDesc bc_desc1;
    TensorDesc bc_desc2;
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

    int n_stacks =
        std::accumulate(desc_.shape.rbegin() + 2, desc_.shape.rend(), 1,
                        std::multiplies<>());

    const int rows = *(bc_desc1.shape.rbegin() + 1);
    const int common = *bc_desc1.shape.rbegin();
    const int cols = *bc_desc2.shape.rbegin();

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
  TensorDesc desc1_;
  TensorDesc desc2_;

  // std::vector<int> calc_output_shape(std::vector<int>& shape1,
  //                                    std::vector<int>& shape2);
};

}  // namespace abyss::core

#endif