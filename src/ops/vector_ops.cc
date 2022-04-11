#include "vector_ops.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

// #include "types.h"

namespace abyss::core {

std::vector<int> VectorVisitor::calc_output_shape(std::vector<int> shape1,
                                                  std::vector<int> shape2) {
  size_t max_dim = std::max(shape1.size(), shape2.size());
  std::vector<int> output_shape(max_dim);
  // prepend ones to match the dimension
  shape1.insert(shape1.begin(), max_dim - shape1.size(), 1);
  shape2.insert(shape2.begin(), max_dim - shape2.size(), 1);

  auto iit1 = shape1.rbegin();
  auto iit2 = shape2.rbegin();
  auto oit = output_shape.rbegin();

  while (oit != output_shape.rend()) {
    // set broadcast flags
    if (*iit1 != *iit2 && !(*iit1 == 1 || *iit2 == 1)) {
      throw std::runtime_error("shapes are not broadcastable");
    }

    // calc shape
    *oit = std::max(*iit1, *iit2);

    iit1++;
    iit2++;
    oit++;
  }

  return output_shape;
}

/**
 * AddVisitor Implementation
 */
AddVisitor::AddVisitor(std::vector<int> shape1, std::vector<int> shape2)
    : VectorVisitor(shape1, shape2) {}

void AddVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<int32_t>* b) {
  eval(a, b, backend::add);
}
void AddVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<double>* b) {
  eval(a, b, backend::add);
}
void AddVisitor::visit(ArrayImpl<double>* a, ArrayImpl<int32_t>* b) {
  eval(a, b, backend::add);
}
void AddVisitor::visit(ArrayImpl<double>* a, ArrayImpl<double>* b) {
  eval(a, b, backend::add);
}

/**
 * SubtractVisitor Implementation
 */
SubtractVisitor::SubtractVisitor(std::vector<int> shape1,
                                 std::vector<int> shape2)
    : VectorVisitor(shape1, shape2) {}

void SubtractVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<int32_t>* b) {
  eval(a, b, backend::sub);
}
void SubtractVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<double>* b) {
  eval(a, b, backend::sub);
}
void SubtractVisitor::visit(ArrayImpl<double>* a, ArrayImpl<int32_t>* b) {
  eval(a, b, backend::sub);
}
void SubtractVisitor::visit(ArrayImpl<double>* a, ArrayImpl<double>* b) {
  eval(a, b, backend::sub);
}

/**
 * MultiplyVisitor Implementation
 */
MultiplyVisitor::MultiplyVisitor(std::vector<int> shape1,
                                 std::vector<int> shape2)
    : VectorVisitor(shape1, shape2) {}

void MultiplyVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<int32_t>* b) {
  eval(a, b, backend::mult);
}
void MultiplyVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<double>* b) {
  eval(a, b, backend::mult);
}
void MultiplyVisitor::visit(ArrayImpl<double>* a, ArrayImpl<int32_t>* b) {
  eval(a, b, backend::mult);
}
void MultiplyVisitor::visit(ArrayImpl<double>* a, ArrayImpl<double>* b) {
  eval(a, b, backend::mult);
}

/**
 * DivideVisitor Implementation
 */
DivideVisitor::DivideVisitor(std::vector<int> shape1, std::vector<int> shape2)
    : VectorVisitor(shape1, shape2) {}

void DivideVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<int32_t>* b) {
  eval(a, b, backend::div);
}
void DivideVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<double>* b) {
  eval(a, b, backend::div);
}
void DivideVisitor::visit(ArrayImpl<double>* a, ArrayImpl<int32_t>* b) {
  eval(a, b, backend::div);
}
void DivideVisitor::visit(ArrayImpl<double>* a, ArrayImpl<double>* b) {
  eval(a, b, backend::div);
}

EqualVisitor::EqualVisitor(std::vector<int> shape1, std::vector<int> shape2)
    : VectorVisitor(shape1, shape2) {}

void EqualVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<int32_t>* b) {
  eval<int32_t, int32_t, bool>(a, b, backend::equal);
}
void EqualVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<double>* b) {
  using T1 = int32_t;
  using T2 = double;
  eval<T1, T2, bool>(a, b, backend::equal);
}
void EqualVisitor::visit(ArrayImpl<double>* a, ArrayImpl<int32_t>* b) {
  eval<double, int32_t, bool>(a, b, backend::equal);
}
void EqualVisitor::visit(ArrayImpl<double>* a, ArrayImpl<double>* b) {
  eval<double, double, bool>(a, b, backend::equal);
}

NotEqualVisitor::NotEqualVisitor(std::vector<int> shape1,
                                 std::vector<int> shape2)
    : VectorVisitor(shape1, shape2) {}

void NotEqualVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<int32_t>* b) {
  eval<int32_t, int32_t, bool>(a, b, backend::not_equal);
}
void NotEqualVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<double>* b) {
  eval<int32_t, double, bool>(a, b, backend::not_equal);
}
void NotEqualVisitor::visit(ArrayImpl<double>* a, ArrayImpl<int32_t>* b) {
  eval<double, int32_t, bool>(a, b, backend::not_equal);
}
void NotEqualVisitor::visit(ArrayImpl<double>* a, ArrayImpl<double>* b) {
  eval<double, double, bool>(a, b, backend::not_equal);
}

}  // namespace abyss::core