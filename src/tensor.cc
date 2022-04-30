#include "tensor.h"

#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <typeindex>
#include <unordered_map>
#include <utility>

#include "core/array.h"
// #include "ops/dtype_ops.h"
#include "ops/conversion_ops.h"
#include "ops/dispatcher.h"
#include "ops/merge_ops.h"
#include "ops/util_ops.h"
#include "traits.h"

namespace abyss {

Tensor::Tensor(const Tensor& other)
    : dtype_{other.dtype_},
      desc_{other.desc_},
      data_{other.data_},
      flags_{other.flags_} {
  // unset editable flag when copied
  flags_ = flags_ & ~TensorFlags::kIsEditable;
}

Tensor::Tensor(Tensor&& other)
    : dtype_{other.dtype_},
      desc_{other.desc_},
      data_{other.data_},
      flags_{other.flags_} {
  // unset editable flag when copied
  // flags_ = flags_ & ~TensorFlags::kIsEditable;
}

Tensor& Tensor::operator=(Tensor copy) {
  if (flags(TensorFlags::kIsEditable)) {
    core::AssignToViewVisitor assign_to_view(copy.desc_, desc_);

    copy.data_->accept(&assign_to_view, data_.get());
  } else {
    swap(copy);
  }
  return *this;
}

Tensor Tensor::copy() {
  core::CopyVisitor copy_visitor(desc_);
  data_->accept(&copy_visitor);

  return copy_visitor;
}

Tensor Tensor::all(int axis) const {
  core::AllVisitor all_visitor(desc_, axis);
  data_->accept(&all_visitor);

  return all_visitor;
}

Tensor Tensor::all() const {
  core::AllVisitor all_visitor(desc_);
  data_->accept(&all_visitor);

  return all_visitor;
}

ScalarType Tensor::dtype() const { return dtype_; }
core::Array* Tensor::data() const { return data_.get(); }
core::TensorDesc Tensor::desc() const { return desc_; }

bool Tensor::flags(core::TensorFlags name) {
  using T = std::underlying_type_t<core::TensorFlags>;
  return static_cast<T>(flags_ & name);
}

size_t Tensor::offset() const { return desc_.offset; }
const std::vector<int>& Tensor::shape() const { return desc_.shape; }
const std::vector<int>& Tensor::strides() const { return desc_.strides; }
size_t Tensor::size() const { return core::shape2size(desc_.shape); }
size_t Tensor::nbytes() const { return dtype_.itemsize() * size(); }
size_t Tensor::ndims() const { return desc_.shape.size(); }

int Tensor::shape(int index) const {
  auto it = index >= 0 ? shape().begin() : shape().end();
  // length check
  return *(it + index);
}

int Tensor::strides(int index) const {
  auto it = index >= 0 ? strides().begin() : strides().end();
  // length check
  return *(it + index);
}

Tensor& Tensor::reshape(std::vector<int> new_shape) {
  if (core::shape2size(new_shape) != size())
    throw std::runtime_error("new shape does not match size");

  init_view();
  /// @bug reshape might not be a view if the data is already not contiguous
  view_->flags_ = TensorFlags::kIsView;
  view_->desc_.shape = new_shape;
  view_->desc_.strides = core::shape2strides(new_shape);

  if (!flags(TensorFlags::kIsContiguous)) {
    core::AssignToViewVisitor assign_to_view(desc_, view_->desc_);

    data_->accept(&assign_to_view, view_->data());
  }

  // return std::move(*view_);
  return *view_;
}

Tensor& Tensor::broadcast_to(std::vector<int> new_shape) {
  if (!core::is_broadcastable(shape(), new_shape)) {
    throw std::domain_error("unable to broadcast to shape");
  }

  init_view();
  view_->flags_ = TensorFlags::kIsView;
  view_->desc_.shape = new_shape;
  view_->desc_.strides = desc_.strides;

  std::vector<int>& new_strides = view_->desc_.strides;
  // new_strides.clear();

  // calculate new strides
  size_t extra_dim = new_shape.size() - desc_.shape.size();
  if (extra_dim > 0) {
    // extra dimensions require broadcast, padd zeros to the front of the
    // strides
    new_strides.insert(new_strides.begin(), extra_dim, 0);
  }

  // set strides of broadcastable axis to 0
  auto it = shape().rbegin();
  auto strides_it = new_strides.rbegin();
  auto d_it = new_shape.rbegin();
  for (size_t i = 0; i < shape().size(); i++) {
    if (*d_it / *it != 1 && *it == 1) {
      *strides_it = 0;
    }

    it++;
    strides_it++;
    d_it++;
  }

  // return std::move(*view_);
  return *view_;
}
Tensor::const_iterator Tensor::begin() const {
  return const_iterator(*this, 0);
}
Tensor::const_iterator Tensor::end() const {
  return const_iterator(*this, shape(0));
}
Tensor::iterator Tensor::begin() { return iterator(*this, 0); }
Tensor::iterator Tensor::end() { return iterator(*this, shape(0)); }

std::ostream& operator<<(std::ostream& os, Tensor tensor) {
  core::Dispatcher<Tensor> tsr(tensor);
  core::ArrayPrintVisitor print_visitor(tsr.desc());

  core::Dispatcher<Tensor> dispatcher(tensor);

  dispatcher.accept(&print_visitor);

  os << print_visitor.str();

  return os;
}

void Tensor::init_view() {
  if (view_ == nullptr) {
    // std::cout<<"view empty" <<std::endl;
    view_ = std::make_unique<Tensor>(*this);
  }

  // clear shapes and strides
  // reset offset back to the offset of the current class
  view_->desc_.offset = desc_.offset;
  view_->desc_.shape.clear();
  view_->desc_.strides.clear();
  view_->flags_ = TensorFlags::kNoFlags;
}


/**
 * Tensor::Iterator implementation
 */
// template <typename T>
Tensor::Iterator::Iterator(const Tensor& self, int index)
    : index_{index}, stride_{self.strides(0)}, offset_{self.offset()}, view_{self} {
  view_.desc_.shape.assign(self.desc_.shape.begin() + 1,
                                  self.desc_.shape.end());
  view_.desc_.strides.assign(self.desc_.strides.begin() + 1,
                                    self.desc_.strides.end());
  view_.flags_ = TensorFlags::kIsView | TensorFlags::kIsEditable;
}

// template <typename T>
Tensor::Iterator& Tensor::Iterator::operator=(Iterator copy) {
  swap(copy);

  return *this;
}

// template <typename T>
Tensor::Iterator& Tensor::Iterator::operator+=(int n) {
  index_ += n;
  return *this;
}
// template <typename T>
Tensor::Iterator& Tensor::Iterator::operator-=(int n) {
  index_ -= n;
  return *this;
}

// template <typename T>
void Tensor::Iterator::swap(Iterator& other) noexcept {
  using std::swap;

  // swap(self_, other.self_);
  swap(index_,other.index_);
  swap(const_cast<int&>(stride_), const_cast<int&>(other.stride_));
  swap(const_cast<size_t&>(offset_), const_cast<size_t&>(other.offset_));
  swap(view_, other.view_);
}

// template <typename T>
Tensor::Iterator::reference Tensor::Iterator::operator*() {
  update_offset(index_);
  return view_;
}
// template <typename T>
Tensor::Iterator::pointer Tensor::Iterator::operator->() {
  update_offset(index_);
  return &view_;
}

// template <typename T>
Tensor::Iterator& Tensor::Iterator::operator++() {
  ++index_;
  return *this;
}
// template <typename T>
Tensor::Iterator& Tensor::Iterator::operator++(int ignore) {
  // return ++*this;
  index_++;
  return *this;
}
// template <typename T>
Tensor::Iterator& Tensor::Iterator::operator--() {
  --index_;
  return *this;
}
// template <typename T>
Tensor::Iterator& Tensor::Iterator::operator--(int ignore) { 
  index_--;
  return *this;
}

// template <typename T>
Tensor::Iterator Tensor::Iterator::operator+(int offset) {
  update_offset(index_);
  return Iterator(view_, index_ + offset);
}
// template <typename T>
Tensor::Iterator Tensor::Iterator::operator-(int offset) {
  update_offset(index_);
  return Iterator(view_, index_ - offset);
}

// template <typename T>
Tensor::Iterator::difference_type Tensor::Iterator::operator-(
    const Iterator& other) {
  return index_ - other.index_;
}

// template <typename T>
Tensor::Iterator::reference Tensor::Iterator::operator[](int index) {
  update_offset(index_ + index);
  return view_;
}

// template <typename T>
bool Tensor::Iterator::operator==(const Iterator& other) const {
  return view_.data() == other.view_.data() && index_ == other.index_;
}
// template <typename T>
bool Tensor::Iterator::operator!=(const Iterator& other) const {
  return view_.data() == other.view_.data() && index_ != other.index_;
}
// template <typename T>
bool Tensor::Iterator::operator>(const Iterator& other) const {
  return view_.data() == other.view_.data() && index_ > other.index_;
}
// template <typename T>
bool Tensor::Iterator::operator<=(const Iterator& other) const {
  return view_.data() == other.view_.data() && index_ <= other.index_;
}
bool Tensor::Iterator::operator>=(const Iterator& other) const {
  return view_.data() == other.view_.data() && index_ >= other.index_;
}
bool Tensor::Iterator::operator<(const Iterator& other) const {
  return view_.data() == other.view_.data() && index_ < other.index_;
}

void Tensor::Iterator::update_offset(int index) {
  // auto coords = core::unravel_index(index, self_.shape());

  size_t offset = offset_ + index * stride_;

  view_.desc_.offset = offset;
}


namespace x {
// experimental implementations

}

}  // namespace abyss
