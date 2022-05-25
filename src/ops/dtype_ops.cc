#include "dtype_ops.h"

namespace abyss::core {
/**
 * EmptyVisitor Implementation
 */

EmptyVisitor::EmptyVisitor(std::vector<int> shape) : shape_{shape} {}

void EmptyVisitor::visit(DTypeImpl<bool>* dtype) {
  eval(dtype);
}
void EmptyVisitor::visit(DTypeImpl<int32_t>* dtype) {
  eval(dtype);
}
void EmptyVisitor::visit(DTypeImpl<double>* dtype) {
  eval(dtype);
}

/**
 * FullVisitor Implementation
 */

FullVisitor::FullVisitor(std::vector<int> shape) : shape_{shape} {}
void FullVisitor::visit(ArrayImpl<bool>* value, DTypeImpl<bool>* dtype) {
  eval(value, dtype);
}
void FullVisitor::visit(ArrayImpl<int32_t>* value, DTypeImpl<bool>* dtype) {
  eval(value, dtype);
}
void FullVisitor::visit(ArrayImpl<int32_t>* value, DTypeImpl<int32_t>* dtype) {
  eval(value, dtype);
}
void FullVisitor::visit(ArrayImpl<double>* value, DTypeImpl<int32_t>* dtype) {
  eval(value, dtype);
}
void FullVisitor::visit(ArrayImpl<int32_t>* value, DTypeImpl<double>* dtype) {
  eval(value, dtype);
}
void FullVisitor::visit(ArrayImpl<double>* value, DTypeImpl<double>* dtype) {
  eval(value, dtype);
}

/**
 * ArangeVisitor Implementation
 */

// void ArangeVisitor::visit(DTypeImpl<bool>* dtype) {
//   eval(dtype);
// }
// void ArangeVisitor::visit(DTypeImpl<uint8_t>* dtype) {
//   eval(dtype);
// }
// void ArangeVisitor::visit(DTypeImpl<int32_t>* dtype) {
//   eval(dtype);
// }
// void ArangeVisitor::visit(DTypeImpl<double>* dtype) {
//   eval(dtype);
// }

  RandNormalVisitor::RandNormalVisitor(std::vector<int> shape) : shape_{shape} {}
  void RandNormalVisitor::visit(DTypeImpl<double>* dtype) { eval(dtype); }

}  // namespace abyss::core