#include "operators.h"

#include "functional.h"
#include "ops/vector_ops.h"

namespace abyss {
Tensor operator+(Tensor a, Tensor b) { return add(a, b); }
Tensor operator-(Tensor a, Tensor b) { return subtract(a, b); }
Tensor operator*(Tensor a, Tensor b) { return multiply(a, b); }
Tensor operator/(Tensor a, Tensor b) { return divide(a, b); }

Tensor operator==(Tensor a, Tensor b) {
  /// @warning this does not work with slices
  /// strides should also be taken into consideration
  using namespace core;
  EqualVisitor equal_visitor(a.shape(), b.shape());

  a.data()->accept(&equal_visitor, b.data());

  return equal_visitor;
}
Tensor operator!=(Tensor a, Tensor b) {
  using namespace core;
  NotEqualVisitor not_equal_visitor(a.shape(), b.shape());

  a.data()->accept(&not_equal_visitor, b.data());

  return not_equal_visitor;
}

}  // namespace abyss
