#ifndef ABYSS_CORE_MATRIX_OPS_H
#define ABYSS_CORE_MATRIX_OPS_H

#include <tuple>
#include <vector>

#include "core/array.h"
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
  MatmulVisitor(std::vector<int> shape1, std::vector<int> shape2);

  void visit(ArrayImpl<int32_t>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<int32_t>*, ArrayImpl<double>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*, ArrayImpl<double>*) override;

  template <typename T1, typename T2>
  void eval(ArrayImpl<T1>* a, ArrayImpl<T2>* b) {
    using result_t = std::common_type_t<T1, T2>;
    std::vector<int> bc_shp1;
    std::vector<int> bc_shp2;
    std::tie(shape_, bc_shp1, bc_shp2) =
        calc_output_shape(shape1_, shape2_);

    // broadcast as needed
    ArrayImpl<T1> a_matched(shape2size(bc_shp1));
    broadcast_copy(a->begin(), a->end(), shape1_, a_matched.begin(), bc_shp1);
    ArrayImpl<T2> b_matched(shape2size(bc_shp2));
    broadcast_copy(b->begin(), b->end(), shape2_, b_matched.begin(), bc_shp2);

    size_t output_size = shape2size(shape_);
    auto c = std::make_shared<ArrayImpl<result_t>>(output_size);

    int n_stacks =
        std::accumulate(shape_.rbegin() + 2, shape_.rend(), 1,
                        std::multiplies<>());

    const int rows = *(bc_shp1.rbegin() + 1);
    const int common = *bc_shp1.rbegin();
    const int cols = *bc_shp2.rbegin();

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
    strides_ = shape2strides(shape_);
    data_ = c;
  }

 private:
  std::vector<int> shape1_;
  std::vector<int> shape2_;

  // std::vector<int> calc_output_shape(std::vector<int>& shape1,
  //                                    std::vector<int>& shape2);
};

}  // namespace abyss::core

#endif