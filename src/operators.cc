#include "operators.h"

#include "functional.h"
#include "core/dispatcher.h"
#include "ops/vector_ops.h"

namespace abyss {
Tensor operator+(Tensor a, Tensor b) { return add(a, b); }
Tensor operator-(Tensor a, Tensor b) { return subtract(a, b); }
Tensor operator*(Tensor a, Tensor b) { return multiply(a, b); }
Tensor operator/(Tensor a, Tensor b) { return divide(a, b); }

Tensor operator==(Tensor lhs, Tensor rhs) {
  /// @warning this does not work with slices
  /// strides should also be taken into consideration
  using namespace core;
  
  DataDispatcher<Tensor> a(lhs);
  DataDispatcher<Tensor> b(rhs);

  EqualVisitor equal_visitor(a.desc(), b.desc());

  a.accept(&equal_visitor, &b);

  return equal_visitor;
}
Tensor operator!=(Tensor lhs, Tensor rhs) {
  using namespace core;  

  DataDispatcher<Tensor> a(lhs);
  DataDispatcher<Tensor> b(rhs);

  NotEqualVisitor not_equal_visitor(a.desc(), b.desc());

  a.accept(&not_equal_visitor, &b);

  return not_equal_visitor;
}

}  // namespace abyss
