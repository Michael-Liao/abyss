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

void ArangeVisitor::visit(ArrayImpl<int32_t>* a, DTypeImpl<int32_t>* dtype) {
  eval(a, dtype);
}
void ArangeVisitor::visit(ArrayImpl<int32_t>* a, DTypeImpl<double>* dtype) {
  eval(a, dtype);
}
void ArangeVisitor::visit(ArrayImpl<double>* a, DTypeImpl<int32_t>* dtype) {
  eval(a, dtype);
}
void ArangeVisitor::visit(ArrayImpl<double>* a, DTypeImpl<double>* dtype) {
  eval(a, dtype);
}

}  // namespace abyss::core