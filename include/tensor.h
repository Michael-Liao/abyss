#ifndef ABYSS_NDARRAY_H
#define ABYSS_NDARRAY_H

/**
 * ref: https://fossies.org/linux/tensorflow/tensorflow/cc/framework/ops.h
 */

#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
// #include <typeindex>
// #include <typeinfo>
// #include <unordered_map>
#include <utility>
// #include <array>
#include <vector>

#include "abyss_export.h"
// #include "types.h"
#include "core/array.h"
#include "core/traits.h"
#include "core/utility.h"
#include "index.h"
#include "scalartype.h"
// #include "core/visitor.h"
// #include "ops/conversion_ops.h"

namespace abyss {

using TensorFlags = core::TensorFlags;

namespace autograd {
// meta graph class
class Graph;

// forward function mixin
template <typename ChildType>
class Function;

// backward function
class BackwardFn;
}  // namespace autograd

/**
 * @brief A type erased multi-dimensional array.
 */
class ABYSS_EXPORT Tensor {
 public:
  // template <typename T>
  // class Initializer;

  // hash needs internal members
  friend struct std::hash<Tensor>;

  // graph create grad during backward ops
  friend class autograd::Graph;

  // autograd function creates grad_fn on the fly
  template <typename ChildType>
  friend class autograd::Function;

  // stashing iterator
  class Iterator;
  // class ReverseIterator;

  using iterator = Iterator;
  using const_iterator = Iterator;

  class KeyEqual;

  Tensor() = default;

  /**
   * @brief Construct Tensor from a nested initializer list.
   *
   * convert initializer list shapes and data into tensors.
   *
   * @tparam T data type that the initializer should be evaluated to.
   */
  // template <typename T>
  // Tensor(Initializer<T> inits);

  /**
   * @brief Construct Tensor from a scalar.
   */
  template <typename T,
            std::enable_if_t<core::is_supported_dtype<T>::value, bool> = true>
  Tensor(T scalar) : dtype_(stypeof<T>(scalar)), desc_{0, {1}, {1}} {
    data_ = std::make_shared<core::ArrayImpl<T>>(1, scalar);
  }

  // all members are copyable and movable, set to default
  // tensor copy constructor is a shallow copy
  Tensor(const Tensor& other);
  Tensor(Tensor&& other);

  Tensor& operator=(Tensor copy);

  virtual ~Tensor() {
    // std::cout << "Tensor destruct" << std::endl;
  }

  void swap(Tensor& other) noexcept {
    using std::swap;

    swap(dtype_, other.dtype_);
    swap(desc_, other.desc_);
    swap(data_, other.data_);
    swap(flags_, other.flags_);
    swap(grad_, other.grad_);
    swap(grad_fn_, other.grad_fn_);
  }

  /**
   * @brief Deep copy of tensors
   */
  Tensor copy();

  /**
   * @brief all elements evaluates to true
   */
  Tensor all(int axis) const;
  Tensor all() const;
  /**
   * @brief any elements evaluates to true
   */
  // Tensor any(int axis) const;
  // Tensor any() const;

  ScalarType dtype() const;
  // core::ArrayDesc desc() const { return desc_; }
  size_t offset() const;
  const std::vector<int>& shape() const;
  const std::vector<int>& strides() const;
  // core::Array* data() const;

  // bool flags(core::TensorFlags name);
  bool flags(core::FlagId name) const;
  void set_flag(core::FlagId name, bool value);

  size_t size() const;
  size_t nbytes() const;
  size_t ndims() const;

  int shape(int index) const;
  int strides(int index) const;
  Tensor& reshape(std::vector<int> new_shape);
  Tensor& broadcast_to(std::vector<int> new_shape);

  const_iterator begin() const;
  const_iterator end() const;
  iterator begin();
  iterator end();

  /**
   * interaction with static types.
   *
   * The conversion operator only converts to what the dtype specifies. Just
   * like std::any_cast
   */
  template <typename T,
            std::enable_if_t<core::is_supported_dtype<T>::value, bool> = true>
  operator T() {
    // static_assert(core::is_supported_dtype<T>::value, "cannot convert to
    // tensor unsupported native scalar");
    if (desc_.shape.size() != 1 || desc_.shape[0] != 1) {
      throw std::domain_error(
          "array of element more than one cannot be converted to scalar");
    }

    if (dtype_.id() != typeid(T)) {
      throw std::runtime_error("conversion to scalar must be the same type");
    }

    auto typed_data = std::dynamic_pointer_cast<core::ArrayImpl<T>>(data_);

    return *typed_data->nbegin(desc_);
  }

  /**
   * @brief slicing using the call operator
   *
   * becuase operator[] does not take more than one argument
   */
  template <typename... IdTypes>
  Tensor& operator()(IdTypes... indices);

  /**
   * @brief tensor transpose
   *
   * https://stackoverflow.com/questions/32034237/how-does-numpys-transpose-method-permute-the-axes-of-an-array
   */
  Tensor& transpose(std::vector<int> axes);

  /**
   * @brief covenient transpose method
   *
   * This returns the most common type of transpose with strides inverted.
   */
  Tensor& T();

  /**
   * @brief funcitons for autograd
   */
  // bool requires_grad();
  // bool is_leaf();
  Tensor& grad();
  autograd::BackwardFn& grad_fn();

  void backward(Tensor gradient = 1);

 protected:
  ScalarType dtype_ = kNone;
  core::ArrayDesc desc_;
  std::shared_ptr<core::Array> data_;
  // TensorFlags flags_ = TensorFlags::kIsContiguous | TensorFlags::kOwnsData;
  TensorFlags flags_;

  std::unique_ptr<Tensor> view_;

  // back-propagation related fields
  // bool requires_grad_ = false;
  std::shared_ptr<Tensor> grad_;
  std::shared_ptr<autograd::BackwardFn> grad_fn_;

  core::Array* data() const;
  core::ArrayDesc desc() const;

  /**
   * @brief initialize view object for writing.
   *
   * allocate view if not exist, and clear flags and tensor description if
   * already allocated.
   */
  void init_view();
  void init_grad();
};

// const Tensor NoData;

// Tensor operator==(const Tensor& a, const Tensor& b);
ABYSS_EXPORT std::ostream& operator<<(std::ostream& os, Tensor tensor);

/**
 * @brief Tensor initializer.
 *
 * A initializer that can initializer a tensor with the correct shape and data
 * from nested `std::initializer_list`s.
 *
 * @tparam T data type to be evaluated.
 */
// template <typename T>
// class Tensor::Initializer {
//  public:
//   // template <typename std::enable_if_t<is_supported_dtype<T>::value, bool>
//   =
//   // true>
//   // template <typename T>
//   Initializer(T scalar);
//   // template <typename std::enable_if_t<is_supported_dtype<T>::value, bool>
//   =
//   // true>
//   // template <typename T>
//   Initializer(std::initializer_list<T> list);

//   Initializer(std::initializer_list<Initializer> lists);

//   std::vector<int> shape() const { return shape_; }
//   std::vector<T> arr() const { return arr_; }

//   // operator core::ArrayImpl<T>() const {
//   //   return core::ArrayImpl<T>(shape_, arr_);
//   // }

//  private:
//   std::vector<int> shape_;
//   std::vector<T> arr_;
// };

/**
 * @brief numpy-like stashing iterator.
 */
// template <typename T>
class Tensor::Iterator {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = Tensor;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::random_access_iterator_tag;

  Iterator(const Tensor& self, int index = 0);

  Iterator(const Iterator&) = default;
  Iterator(Iterator&&) = default;

  Iterator& operator=(Iterator copy);

  Iterator& operator+=(int n);
  Iterator& operator-=(int n);

  ~Iterator() = default;

  void swap(Iterator& other) noexcept;

  reference operator*();
  pointer operator->();

  Iterator& operator++();
  Iterator& operator++(int ignore);
  Iterator& operator--();
  Iterator& operator--(int ignore);

  Iterator operator+(int offset);
  Iterator operator-(int offset);

  difference_type operator-(const Iterator& other);

  reference operator[](int index);

  bool operator==(const Iterator& other) const;
  bool operator!=(const Iterator& other) const;
  bool operator>(const Iterator& other) const;
  bool operator<=(const Iterator& other) const;
  bool operator>=(const Iterator& other) const;
  bool operator<(const Iterator& other) const;

 private:
  // std::shared_ptr<core::Array> ptr_;
  // Tensor self_;
  int index_ = 0;  // index to unravel
  const int stride_ = 0;
  const size_t offset_ = 0;

  Tensor view_;

  /**
   * @brief convert index into offset and assign to view
   */
  void update_offset(int index);
};

class Tensor::KeyEqual {
 public:
  bool operator()(const Tensor& a, const Tensor& b) const {
    return a.data_ == b.data_;
  }
};

/**
 * Tensor implementations
 */

template <typename... IdTypes>
Tensor& Tensor::operator()(IdTypes... indices) {
  constexpr int kIdSize = sizeof...(IdTypes);
  if (ndims() < kIdSize) {
    std::domain_error("too many slices");
  }
  std::array<Index, kIdSize> ids{std::forward<Index>(indices)...};

  init_view();
  // view_->flags_ = TensorFlags::kIsView | TensorFlags::kIsEditable;
  view_->set_flag(core::FlagId::kIsView, true);
  view_->set_flag(core::FlagId::kIsEditable, true);

  bool is_contiguous = true;
  auto shape_it = desc_.shape.begin();
  auto strides_it = desc_.strides.begin();
  for (size_t i = 0; i < ids.size(); i++) {
    view_->desc_.offset += *strides_it * ids[i].start;
    int size = std::min(*shape_it, ids[i].stop - ids[i].start);
    if (size != 1) {
      is_contiguous &= (ids[i].step == 1);
      view_->desc_.shape.emplace_back(size);
      view_->desc_.strides.emplace_back(*strides_it * ids[i].step);
    }

    shape_it++;
    strides_it++;
  }

  // extra dimensions without index will be passed on as well
  while (shape_it != desc_.shape.end()) {
    view_->desc_.shape.emplace_back(*shape_it);
    view_->desc_.strides.emplace_back(*strides_it);

    shape_it++;
    strides_it++;
  }

  // is a scalar
  if (view_->desc_.shape.empty()) {
    view_->desc_.shape.emplace_back(1);
    view_->desc_.strides.emplace_back(1);
  }

  if (is_contiguous) {
    // view_->flags_ = view_->flags_ | TensorFlags::kIsContiguous;
    view_->set_flag(core::FlagId::kIsContiguous, true);
  }

  // return std::move(*view_);
  return *view_;
}

// template <typename T>
// Tensor::Tensor(Tensor::Initializer<T> inits)
//     : dtype_{stypeof<T>()}, shape_{inits.shape()} {
//   strides_ = core::shape2strides(shape_);
//   data_ = std::make_shared<core::ArrayImpl<T>>(inits.shape(), inits.arr());
// }

// template <typename T>
// Tensor array(Tensor::Initializer<T> obj) {
//   return Tensor(obj);
// }

/**
 * Tensor::Initializer implementations
 */
// template <typename T>
// // template <typename std::enable_if_t<is_supported_dtype<T>::value, bool> =
// // true>
// Tensor::Initializer<T>::Initializer(T scalar) : shape_{1}, arr_{scalar} {}

// template <typename T>
// // template <typename std::enable_if_t<is_supported_dtype<T>::value, bool> =
// // true>
// Tensor::Initializer<T>::Initializer(std::initializer_list<T> list)
//     : shape_{static_cast<int>(list.size())}, arr_{list} {}

// template <typename T>
// Tensor::Initializer<T>::Initializer(std::initializer_list<Initializer<T>>
// lists)
//     : shape_{static_cast<int>(lists.size())} {
//   std::vector<int> ref_shape;
//   bool first = true;
//   for (auto& inits : lists) {
//     if (first) {
//       ref_shape.assign(inits.shape_.begin(), inits.shape_.end());
//       std::copy(ref_shape.begin(), ref_shape.end(),
//       std::back_inserter(shape_)); first = false; continue;
//     }

//     // check shapes
//     if (inits.shape_ != ref_shape) {
//       std::runtime_error("Initializer: shape in each dimension must match.");
//     }
//     std::copy(inits.arr_.begin(), inits.arr_.end(),
//     std::back_inserter(arr_));
//   }
// }

}  // namespace abyss

/**
 * @brief make Tensor hashable
 */
template <>
struct std::hash<abyss::Tensor> {
  size_t operator()(const abyss::Tensor& arr) const noexcept {
    bool requires_grad = arr.flags(abyss::core::FlagId::kRequiresGrad);
    return std::hash<abyss::core::Array*>{}(arr.data_.get()) ^ (std::hash<bool>{}(requires_grad) << 1);
    // return arr.data_.get();
  }
};

#endif  // ABYSS_TENSOR_H
