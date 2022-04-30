#include "matrix_ops.h"

#include <cmath>
#include <exception>


namespace abyss::core {

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
MatmulVisitor::calc_output_shape(std::vector<int> shape1, std::vector<int> shape2) {
  if (shape1.size() < 2) {
    shape1.insert(shape1.begin(), 1);
  }
  if (shape2.size() < 2) {
    shape2.emplace_back(1);
  }
  // throws if common shape is not matched
  if (*shape1.rbegin() != *(shape2.rbegin() + 1)) {
    throw std::runtime_error("common shape mismatch");
  }

  size_t max_dim = std::max(shape1.size(), shape2.size());
  std::vector<int> output_shape(max_dim);
  shape1.insert(shape1.begin(), max_dim - shape1.size(), 1);
  shape2.insert(shape2.begin(), max_dim - shape2.size(), 1);

  auto it1 = shape1.rbegin() + 2;
  auto it2 = shape2.rbegin() + 2;
  auto oit = output_shape.rbegin();

  *oit++ = *shape2.rbegin();
  *oit++ = *(shape1.rbegin() + 1);

  while (oit != output_shape.rend()) {
    // throws if shapes are not broadcastable
    if (*it1 != *it2 && !(*it1 == 1 || *it2 == 1)) {
      throw std::runtime_error("matmul: shapes are not broadcastable");
    }

    // calc shape
    const int size = std::max(*it1, *it2);
    oit = std::copy_n(&size, 1, oit);
    it1 = std::copy_n(&size, 1, it1);
    it2 = std::copy_n(&size, 1, it2);
  }

  return std::make_tuple(output_shape, shape1, shape2);
}

MatmulVisitor::MatmulVisitor(TensorDesc desc1, TensorDesc desc2)
    : desc1_{desc1}, desc2_{desc2} {}
void MatmulVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<int32_t>* b) {
  eval(a, b);
}
void MatmulVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<double>* b) {
  eval(a, b);
}
void MatmulVisitor::visit(ArrayImpl<double>* a, ArrayImpl<int32_t>* b) {
  eval(a, b);
}
void MatmulVisitor::visit(ArrayImpl<double>* a, ArrayImpl<double>* b) {
  eval(a, b);
}

}  // namespace abyss::core
