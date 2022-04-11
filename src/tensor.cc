#include "tensor.h"

#include <functional>
#include <numeric>
#include <stdexcept>
#include <typeindex>
#include <unordered_map>
#include <utility>

#include "core/array.h"
// #include "ops/dtype_ops.h"
#include "ops/util_ops.h"
#include "ops/merge_ops.h"
#include "ops/conversion_ops.h"
#include "traits.h"

namespace abyss {

// Tensor Tensor::arange(int start, int stop, int step, ScalarType dtype) {
//   core::ArangeVisitor<int, int, int> arange_visitor(start, stop, step);
//   if (dtype == kNone) {
//     using result_t = std::common_type_t<int, int, int>;
//     dtype = stypeof<result_t>();
//   }
//   dtype->accept(&arange_visitor);

//   return arange_visitor.result();
// }
// Tensor Tensor::arange(int stop, ScalarType dtype) {
//   // aggregated to the first function
//   return arange(0, stop, 1, dtype);
// }

Tensor::Tensor(bool scalar) : dtype_{stypeof(scalar)}, shape_{1}, strides_{1} {
  data_ = std::make_shared<core::ArrayImpl<bool>>(1, scalar);
}
Tensor::Tensor(uint8_t scalar) : dtype_{stypeof(scalar)}, shape_{1}, strides_{1} {
  data_ = std::make_shared<core::ArrayImpl<uint8_t>>(1, scalar);
}
Tensor::Tensor(int32_t scalar) : dtype_{stypeof(scalar)}, shape_{1}, strides_{1} {
  data_ = std::make_shared<core::ArrayImpl<int32_t>>(1, scalar);
}
Tensor::Tensor(double scalar) : dtype_{stypeof(scalar)}, shape_{1}, strides_{1} {
  data_ = std::make_shared<core::ArrayImpl<double>>(1, scalar);
}
Tensor::Tensor(std::complex<double> scalar) : dtype_{stypeof(scalar)}, shape_{1}, strides_{1} {
  data_ = std::make_shared<core::ArrayImpl<std::complex<double>>>(1, scalar);
}

// Tensor::Tensor(ScalarType dtype, std::vector<int> shape,
//                std::shared_ptr<core::Array> data)
//     : dtype_{dtype}, shape_{shape}, strides_{core::shape2strides(shape)} {
//   data_ = data;
// }

Tensor Tensor::copy() {
  core::CopyVisitor copy_visitor(shape_);
  data()->accept(&copy_visitor);

  return copy_visitor;
}

Tensor Tensor::all(int axis) const {
  core::AllVisitor all_visitor(shape_, axis);
  data_->accept(&all_visitor);

  return all_visitor;
}

Tensor Tensor::all() const {
  core::AllVisitor all_visitor(shape_);
  data_->accept(&all_visitor);

  return all_visitor;
}

const ScalarType& Tensor::dtype() const { return dtype_; }
core::Array* Tensor::data() const { return data_.get(); }

const std::vector<int>& Tensor::shape() const { return shape_; }
const std::vector<int>& Tensor::strides() const { return strides_; }
size_t Tensor::size() const { return data_->size(); }
size_t Tensor::nbytes() const { return dtype_->itemsize() * size(); }
size_t Tensor::ndims() const { return shape_.size(); }

int Tensor::shape(int index) const {
  auto it = index >= 0 ? shape().begin() : shape().end();
  // length check
  return *(it + index);
}

Tensor Tensor::reshape(std::vector<int> new_shape) {
  if (core::shape2size(new_shape) != size())
    throw std::runtime_error("new shape does not match size");

  Tensor out;
  out.dtype_ = dtype_;
  out.shape_ = new_shape;
  out.strides_ = core::shape2strides(new_shape);
  out.data_ = data_;

  return out;
}

Tensor::operator bool() {
  core::ToScalarVisitor<bool> to_scalar_vis;
  data_->accept(&to_scalar_vis);

  return to_scalar_vis.value();
}
Tensor::operator uint8_t() {
  core::ToScalarVisitor<uint8_t> to_scalar_vis;
  data_->accept(&to_scalar_vis);

  return to_scalar_vis.value();
}
Tensor::operator int32_t() {
  core::ToScalarVisitor<int32_t> to_scalar_vis;
  data_->accept(&to_scalar_vis);

  return to_scalar_vis.value();  
}
Tensor::operator double() {
  core::ToScalarVisitor<double> to_scalar_vis;
  data_->accept(&to_scalar_vis);

  return to_scalar_vis.value();
}


// Tensor operator==(const Tensor& a, const Tensor& b) {
//   if (a.shape() != b.shape())
//     throw std::runtime_error("cannot comparee 2 tensors with different shape.");

//   core::ComparisonVisitor<std::equal_to<>> comparison_visitor(a.shape());

//   a.data()->accept(&comparison_visitor, b.data());

//   return comparison_visitor.result();
// }

std::ostream& operator<<(std::ostream& os, Tensor tensor) {
  core::ArrayPrintVisitor print_visitor(tensor.shape());

  tensor.data()->accept(&print_visitor);

  os << print_visitor.str();

  return os;
}

namespace x {
// experimental implementations

}

}  // namespace abyss
