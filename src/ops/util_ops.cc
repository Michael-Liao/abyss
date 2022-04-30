#include "util_ops.h"

// #include <memory>
// #include <ostream>

#include "traits.h"

namespace abyss::core {

/**
 * CopyVisitor implementation
 */
CopyVisitor::CopyVisitor(TensorDesc desc) : in_desc_{desc} {}

void CopyVisitor::visit(ArrayImpl<int32_t>* from) {
  eval(from);
}
void CopyVisitor::visit(ArrayImpl<double>* from) {
  eval(from);
}

/**
 * AssignToViewVisitor Implementation
 */
AssignToViewVisitor::AssignToViewVisitor(TensorDesc desc1, TensorDesc desc2)
    : desc1_{desc1}, desc2_{desc2} {}
void AssignToViewVisitor::visit(ArrayImpl<int32_t>* from,
                                ArrayImpl<int32_t>* to) {
  eval(from, to);
}
void AssignToViewVisitor::visit(ArrayImpl<int32_t>* from,
                                ArrayImpl<double>* to) {
  eval(from, to);
}
void AssignToViewVisitor::visit(ArrayImpl<double>* from,
                                ArrayImpl<int32_t>* to) {
  eval(from, to);
}
void AssignToViewVisitor::visit(ArrayImpl<double>* from,
                                ArrayImpl<double>* to) {
  eval(from, to);
}

/**
 * ArrayPrintVisitor Implementations
 */
void ArrayPrintVisitor::visit(ArrayImpl<bool>* a) {
  eval(a);
}
void ArrayPrintVisitor::visit(ArrayImpl<uint8_t>* a) {
  eval(a);
}

void ArrayPrintVisitor::visit(ArrayImpl<int32_t>* a) {
  eval(a);
}
void ArrayPrintVisitor::visit(ArrayImpl<double>* a) {
  eval(a);
}

}  // namespace abyss::core