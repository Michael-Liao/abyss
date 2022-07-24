#include "merge_ops.h"

#include <algorithm>
#include <exception>
#include <functional>
#include <type_traits>

#include "backend/reduction.h"

namespace abyss::core {

std::vector<int> ConcatVisitor::calc_output_shape(std::vector<int> a,
                                                  std::vector<int> b,
                                                  int ignore_axis) {
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i] && i != ignore_axis) {
      throw std::runtime_error("shapes do not match, concat failed.");
    }
  }
  a[ignore_axis] += b[ignore_axis];

  return a;
}

ConcatVisitor::ConcatVisitor(std::vector<int> shape1, std::vector<int> shape2,
                             int axis)
    : shape1_{shape1}, shape2_{shape2}, axis_{axis} {}
void ConcatVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<int32_t>* b) {
  eval(a, b);
}
void ConcatVisitor::visit(ArrayImpl<int32_t>* a, ArrayImpl<double>* b) {
  eval(a, b);
}
void ConcatVisitor::visit(ArrayImpl<double>* a, ArrayImpl<int32_t>* b) {
  eval(a, b);
}
void ConcatVisitor::visit(ArrayImpl<double>* a, ArrayImpl<double>* b) {
  eval(a, b);
}

// bool ConcatVisitor::check_shape(const std::vector<int>& a,
//                                 const std::vector<int>& b, int ignore_axis) {
//   for (size_t i = 0; i < a.size(); i++) {
//     if (a[i] != b[i] && i != ignore_axis) {
//       return false;
//     }
//   }

//   return true;
// }
// std::vector<int> ConcatVisitor::calc_output_shape(const std::vector<int>& a,
//                                                   const std::vector<int>& b,
//                                                   int ignore_axis) {
//   std::vector<int> new_shape = a;
//   new_shape[ignore_axis] += b[ignore_axis];

//   return new_shape;
// }

ReductionVisitor::ReductionVisitor(ArrayDesc desc, int axis)
    : axis_{axis}, in_desc_{desc} {
  // rectify to only positive axis
  if (axis_ < 0) {
    axis_ = desc.shape.size() - axis_;
  }
}

std::vector<int> AllVisitor::calc_output_shape(std::vector<int> shape,
                                               int axis) {
  if (axis == kNoAxis) {
    return {1};
  }
  for (size_t i = 0; i < shape.size(); i++) {
    if (i == axis) shape[i] = 1;
  }

  return shape;
}

AllVisitor::AllVisitor(ArrayDesc desc, int axis)
    : axis_{axis}, in_desc_{desc} {}

void AllVisitor::visit(ArrayImpl<bool>* a) { eval(a); }
void AllVisitor::visit(ArrayImpl<uint8_t>* a) { eval(a); }
void AllVisitor::visit(ArrayImpl<int32_t>* a) { eval(a); }
void AllVisitor::visit(ArrayImpl<double>* a) { eval(a); }

SumVisitor::SumVisitor(ArrayDesc desc) : ReductionVisitor(desc) {}
SumVisitor::SumVisitor(ArrayDesc desc, int axis)
    : ReductionVisitor(desc, axis) {}

void SumVisitor::visit(ArrayImpl<bool>* a) { eval(a, backend::sum); }
void SumVisitor::visit(ArrayImpl<uint8_t>* a) { eval(a, backend::sum); }
void SumVisitor::visit(ArrayImpl<int32_t>* a) { eval(a, backend::sum); }
void SumVisitor::visit(ArrayImpl<double>* a) { eval(a, backend::sum); }

}  // namespace abyss::core