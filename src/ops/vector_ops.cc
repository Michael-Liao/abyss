#include "vector_ops.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

// #include "types.h"
// #include "backend/arr_math.h"

namespace abyss::core {

// std::vector<int> BinaryVectorVisitor::calc_output_shape(std::vector<int> shape1,
//                                                   std::vector<int> shape2) {
//   size_t max_dim = std::max(shape1.size(), shape2.size());
//   std::vector<int> output_shape(max_dim);
//   // prepend ones to match the dimension
//   shape1.insert(shape1.begin(), max_dim - shape1.size(), 1);
//   shape2.insert(shape2.begin(), max_dim - shape2.size(), 1);

//   auto iit1 = shape1.rbegin();
//   auto iit2 = shape2.rbegin();
//   auto oit = output_shape.rbegin();

//   while (oit != output_shape.rend()) {
//     // set broadcast flags
//     if (*iit1 != *iit2 && !(*iit1 == 1 || *iit2 == 1)) {
//       throw std::runtime_error("shapes are not broadcastable");
//     }

//     // calc shape
//     *oit = std::max(*iit1, *iit2);

//     iit1++;
//     iit2++;
//     oit++;
//   }

//   return output_shape;
// }

/**
 * AddVisitor Implementation
 */
AddVisitor::AddVisitor(ArrayDesc desc1, ArrayDesc desc2)
    : BinaryVectorVisitor(desc1, desc2) {}

void AddVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<int32_t>* b) {
  // eval(a, b, backend::add);
  broadcast_eval(a, b, backend::add);
}
void AddVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<double>* b) {
  // eval(a, b, backend::add);
  broadcast_eval(a, b, backend::add);
}
void AddVisitor::visit(ArrayImpl<double>* a, ArrayImpl<int32_t>* b) {
  // eval(a, b, backend::add);
  broadcast_eval(a, b, backend::add);
}
void AddVisitor::visit(ArrayImpl<double>* a, ArrayImpl<double>* b) {
  // eval(a, b, backend::add);
  broadcast_eval(a, b, backend::add);
}

/**
 * SubtractVisitor Implementation
 */
SubtractVisitor::SubtractVisitor(ArrayDesc desc1, ArrayDesc desc2)
    : BinaryVectorVisitor(desc1, desc2) {}

void SubtractVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<int32_t>* b) {
  // eval(a, b, backend::sub);
  broadcast_eval(a, b, backend::sub);
}
void SubtractVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<double>* b) {
  // eval(a, b, backend::sub);
  broadcast_eval(a, b, backend::sub);
}
void SubtractVisitor::visit(ArrayImpl<double>* a, ArrayImpl<int32_t>* b) {
  // eval(a, b, backend::sub);
  broadcast_eval(a, b, backend::sub);
}
void SubtractVisitor::visit(ArrayImpl<double>* a, ArrayImpl<double>* b) {
  // eval(a, b, backend::sub);
  broadcast_eval(a, b, backend::sub);
}

/**
 * MultiplyVisitor Implementation
 */
MultiplyVisitor::MultiplyVisitor(ArrayDesc desc1, ArrayDesc desc2)
    : BinaryVectorVisitor(desc1, desc2) {}

void MultiplyVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<int32_t>* b) {
  // eval(a, b, backend::mult);
  broadcast_eval(a, b, backend::mult);
}
void MultiplyVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<double>* b) {
  // eval(a, b, backend::mult);
  broadcast_eval(a, b, backend::mult);
}
void MultiplyVisitor::visit(ArrayImpl<double>* a, ArrayImpl<int32_t>* b) {
  // eval(a, b, backend::mult);
  broadcast_eval(a, b, backend::mult);
}
void MultiplyVisitor::visit(ArrayImpl<double>* a, ArrayImpl<double>* b) {
  // eval(a, b, backend::mult);
  broadcast_eval(a, b, backend::mult);
}

/**
 * DivideVisitor Implementation
 */
DivideVisitor::DivideVisitor(ArrayDesc desc1, ArrayDesc desc2)
    : BinaryVectorVisitor(desc1, desc2) {}

void DivideVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<int32_t>* b) {
  // eval(a, b, backend::div);
  broadcast_eval(a, b, backend::div);
}
void DivideVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<double>* b) {
  // eval(a, b, backend::div);
  broadcast_eval(a, b, backend::div);
}
void DivideVisitor::visit(ArrayImpl<double>* a, ArrayImpl<int32_t>* b) {
  // eval(a, b, backend::div);
  broadcast_eval(a, b, backend::div);
}
void DivideVisitor::visit(ArrayImpl<double>* a, ArrayImpl<double>* b) {
  // eval(a, b, backend::div);
  broadcast_eval(a, b, backend::div);
}

EqualVisitor::EqualVisitor(ArrayDesc desc1, ArrayDesc desc2)
    : BinaryVectorVisitor(desc1, desc2) {}

void EqualVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<int32_t>* b) {
  // eval<int32_t, int32_t, bool>(a, b, backend::equal);
  broadcast_eval<int32_t, int32_t, bool>(a, b, backend::equal);
}
void EqualVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<double>* b) {
  // eval<int32_t, double, bool>(a, b, backend::equal);
  broadcast_eval<int32_t, double, bool>(a, b, backend::equal);
}
void EqualVisitor::visit(ArrayImpl<double>* a, ArrayImpl<int32_t>* b) {
  // eval<double, int32_t, bool>(a, b, backend::equal);
  broadcast_eval<double, int32_t, bool>(a, b, backend::equal);
}
void EqualVisitor::visit(ArrayImpl<double>* a, ArrayImpl<double>* b) {
  // eval<double, double, bool>(a, b, backend::equal);
  broadcast_eval<double, double, bool>(a, b, backend::equal);
}

NotEqualVisitor::NotEqualVisitor(ArrayDesc desc1, ArrayDesc desc2)
    : BinaryVectorVisitor(desc1, desc2) {}

void NotEqualVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<int32_t>* b) {
  // eval<int32_t, int32_t, bool>(a, b, backend::not_equal);
  broadcast_eval<int32_t, int32_t, bool>(a, b, backend::not_equal);
}
void NotEqualVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<double>* b) {
  // eval<int32_t, double, bool>(a, b, backend::not_equal);
  broadcast_eval<int32_t, double, bool>(a, b, backend::not_equal);
}
void NotEqualVisitor::visit(ArrayImpl<double>* a, ArrayImpl<int32_t>* b) {
  // eval<double, int32_t, bool>(a, b, backend::not_equal);
  broadcast_eval<double, int32_t, bool>(a, b, backend::not_equal);
}
void NotEqualVisitor::visit(ArrayImpl<double>* a, ArrayImpl<double>* b) {
  // eval<double, double, bool>(a, b, backend::not_equal);
  broadcast_eval<double, double, bool>(a, b, backend::not_equal);
}


ExpVisitor::ExpVisitor(ArrayDesc desc) : UnaryVectorVisitor{desc} {}

void ExpVisitor::visit(ArrayImpl<uint8_t>* a) {
  eval(a, backend::exp);
}
void ExpVisitor::visit(ArrayImpl<int32_t>* a) {
  eval(a, backend::exp);
}
void ExpVisitor::visit(ArrayImpl<double>* a) {
  eval(a, backend::exp);
}

NegateVisitor::NegateVisitor(ArrayDesc desc) : UnaryVectorVisitor{desc} {}

void NegateVisitor::visit(ArrayImpl<uint8_t>* a) {
  eval(a, backend::neg);
}
void NegateVisitor::visit(ArrayImpl<int32_t>* a) {
  eval(a, backend::neg);
}
void NegateVisitor::visit(ArrayImpl<double>* a) {
  eval(a, backend::neg);
}

LogVisitor::LogVisitor(ArrayDesc desc) : UnaryVectorVisitor{desc} {}

void LogVisitor::visit(ArrayImpl<uint8_t>* a) {
  eval(a, backend::log);
}
void LogVisitor::visit(ArrayImpl<int32_t>* a) {
  eval(a, backend::log);
}
void LogVisitor::visit(ArrayImpl<double>* a) {
  eval(a, backend::log);
}

}  // namespace abyss::core