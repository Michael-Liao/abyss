#include "functional.h"

#include <algorithm>
#include <exception>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

// #include "types.h"
#include "ops/vector_ops.h"
#include "ops/merge_ops.h"
#include "ops/matrix_ops.h"
#include "ops/dtype_ops.h"

namespace abyss {
Tensor empty(std::vector<int> shape, ScalarType dtype) {
  core::EmptyVisitor empty_visitor(shape);

  dtype->accept(&empty_visitor);

  // return empty_visitor.result();
  return empty_visitor;
}
Tensor full(std::vector<int> shape, Tensor fill_value,
                    ScalarType dtype) {
  core::FullVisitor full_visitor(shape);

  if (dtype == kNone) { // use infered type from fill value
    dtype = fill_value.dtype();
  }

  fill_value.data()->accept(&full_visitor, dtype);

  // return full_visitor.result();
  return full_visitor;
}
Tensor arange(Tensor start, Tensor stop, Tensor step, ScalarType dtype) {
  Tensor bundle = concat({start, stop, step});
  if (dtype == kNone) {
    dtype = bundle.dtype();
  }

  core::ArangeVisitor arange_visitor;

  bundle.data()->accept(&arange_visitor, dtype);

  // return arange_visitor.result();
  return arange_visitor;
}
Tensor arange(Tensor stop, ScalarType dtype) {
  return arange(0, stop, 1, dtype);
}

Tensor concat(std::vector<Tensor> tensors, int axis) {
  if (axis < 0)
    throw std::runtime_error("concat currently supports axis >= 0.");

  // create deep copy from the first one
  Tensor out = tensors[0].copy();
  for (int i = 1; i < tensors.size(); i++) {
    core::ConcatVisitor concat_visitor(out.shape(), tensors[i].shape(), axis);
    out.data()->accept(&concat_visitor, tensors[i].data());
    // assign result back to the output tensor
    out = concat_visitor;
  }

  return out;
}

Tensor add(Tensor lhs, Tensor rhs) {
  using namespace core;
  
  AddVisitor add_visitor(lhs.shape(), rhs.shape());
  // core::AddVisitor add_visitor(lhs.shape(), rhs.shape());
  lhs.data()->accept(&add_visitor, rhs.data());

  return add_visitor;
}

Tensor subtract(Tensor lhs, Tensor rhs) {
  using namespace core;
  
  SubtractVisitor subtract_visitor(lhs.shape(), rhs.shape());
  // core::AddVisitor add_visitor(lhs.shape(), rhs.shape());
  lhs.data()->accept(&subtract_visitor, rhs.data());

  return subtract_visitor;
}

Tensor multiply(Tensor lhs, Tensor rhs) {
  using namespace core;
  
  MultiplyVisitor multiply_visitor(lhs.shape(), rhs.shape());
  // core::AddVisitor add_visitor(lhs.shape(), rhs.shape());
  lhs.data()->accept(&multiply_visitor, rhs.data());

  return multiply_visitor;
}

Tensor divide(Tensor lhs, Tensor rhs) {
  using namespace core;
  
  DivideVisitor divide_visitor(lhs.shape(), rhs.shape());
  // core::AddVisitor add_visitor(lhs.shape(), rhs.shape());
  lhs.data()->accept(&divide_visitor, rhs.data());

  return divide_visitor;
}

Tensor matmul(Tensor lhs, Tensor rhs) {
  core::MatmulVisitor matmul_visitor(lhs.shape(), rhs.shape());

  lhs.data()->accept(&matmul_visitor, rhs.data());

  return matmul_visitor;
}

}  // namespace abyss
