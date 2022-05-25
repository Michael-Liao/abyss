#include "functional.h"

#include <algorithm>
#include <exception>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

// #include "types.h"
// #include "ops/dispatcher.h"
#include "ops/vector_ops.h"
#include "ops/merge_ops.h"
#include "ops/matrix_ops.h"
#include "ops/dtype_ops.h"
#include "autograd/functors.h"

namespace abyss {
Tensor empty(std::vector<int> shape, ScalarType dtype) {
  core::EmptyVisitor empty_visitor(shape);

  core::TypeDispatcher<ScalarType> dtype_dispatch(dtype);

  dtype_dispatch.accept(&empty_visitor);

  // return empty_visitor.result();
  return empty_visitor;
}
Tensor full(std::vector<int> shape, Tensor fill_value,
                    ScalarType dtype) {
  using namespace core;
  
  DataDispatcher<Tensor> scalar = fill_value;

  FullVisitor full_visitor(shape);

  if (dtype == kNone) { // use infered type from fill value
    dtype = fill_value.dtype();
  }
  
  TypeDispatcher<ScalarType> dtype_diispatch = dtype;

  // fill_value.data()->accept(&full_visitor, dtype);
  scalar.accept(&full_visitor, &dtype_diispatch);

  // return full_visitor.result();
  return full_visitor;
}
// Tensor arange(Tensor start, Tensor stop, Tensor step, ScalarType dtype) {
//   Tensor bundle = concat({start, stop, step});
//   if (dtype == kNone) {
//     dtype = bundle.dtype();
//   }

//   core::ArangeVisitor arange_visitor;

//   bundle.data()->accept(&arange_visitor, dtype);

//   // return arange_visitor.result();
//   return arange_visitor;
// }
// Tensor arange(Tensor stop, ScalarType dtype) {
//   return arange(0, stop, 1, dtype);
// }
Tensor randn(std::vector<int> shape, ScalarType dtype) {
  using namespace core;
  if (dtype == kNone) {
    dtype = stypeof<double>();
  }
  TypeDispatcher<ScalarType> stype = dtype;

  RandNormalVisitor randn_vis(shape);
  stype.accept(&randn_vis);
  

  return randn_vis;
}

Tensor concat(std::vector<Tensor> tensors, int axis) {
  if (axis < 0)
    throw std::runtime_error("concat currently supports axis >= 0.");

  using namespace core;

  // create deep copy from the first one
  DataDispatcher<Tensor> out = tensors[0].copy();
  for (int i = 1; i < tensors.size(); i++) {
    ConcatVisitor concat_visitor(out.shape(), tensors[i].shape(), axis);
    // out.data()->accept(&concat_visitor, tensors[i].data());
    DataDispatcher<Tensor> dtsr = tensors[i];
    out.accept(&concat_visitor, &dtsr);
    // assign result back to the output tensor
    out = concat_visitor;
  }

  return out;
}

// Tensor add(Tensor lhs, Tensor rhs) {
//   using namespace core;

//   DataDispatcher<Tensor> a(lhs);
//   DataDispatcher<Tensor> b(rhs);
  
//   // AddVisitor add_visitor(lhs.desc(), rhs.desc());
//   AddVisitor add_visitor(a.desc(), b.desc());
//   // core::AddVisitor add_visitor(lhs.shape(), rhs.shape());
//   // lhs.data()->accept(&add_visitor, rhs.data());

//   // std::cout<<"add function start: ";

//   a.accept(&add_visitor, &b);

//   return add_visitor;
// }
Tensor add(Tensor lhs, Tensor rhs) {
  // using namespace core;
  autograd::AddFn add_fn;

  return add_fn.call(lhs, rhs);
}

Tensor subtract(Tensor lhs, Tensor rhs) {
  using namespace core;

  DataDispatcher<Tensor> a(lhs);
  DataDispatcher<Tensor> b(rhs);
  
  SubtractVisitor subtract_visitor(a.desc(), b.desc());
  // core::AddVisitor add_visitor(lhs.shape(), rhs.shape());
  // lhs.data()->accept(&subtract_visitor, rhs.data());
  a.accept(&subtract_visitor, &b);

  return subtract_visitor;
}

Tensor multiply(Tensor lhs, Tensor rhs) {
  using namespace core;
  
  DataDispatcher<Tensor> a(lhs);
  DataDispatcher<Tensor> b(rhs);
  
  MultiplyVisitor multiply_visitor(a.desc(), b.desc());
  // core::AddVisitor add_visitor(lhs.shape(), rhs.shape());
  a.accept(&multiply_visitor, &b);

  return multiply_visitor;
}

Tensor divide(Tensor lhs, Tensor rhs) {
  using namespace core;
  
  DataDispatcher<Tensor> a(lhs);
  DataDispatcher<Tensor> b(rhs);
  
  DivideVisitor divide_visitor(a.desc(), b.desc());
  // core::AddVisitor add_visitor(lhs.shape(), rhs.shape());
  a.accept(&divide_visitor, &b);

  return divide_visitor;
}

// Tensor matmul(Tensor lhs, Tensor rhs) {
//   using namespace core;

//   DataDispatcher<Tensor> a(lhs);
//   DataDispatcher<Tensor> b(rhs);
  
//   MatmulVisitor matmul_visitor(a.desc(), b.desc());
  
//   // lhs.data()->accept(&matmul_visitor, rhs.data());
//   a.accept(&matmul_visitor, &b);

//   return matmul_visitor;
// }
Tensor matmul(Tensor lhs, Tensor rhs) {
  autograd::MatmulFn matmul_fn;

  return matmul_fn.call(lhs, rhs);
}

}  // namespace abyss
