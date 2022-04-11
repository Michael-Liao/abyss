#ifndef ABYSS_TENSOR_H
#define ABYSS_TENSOR_H

/**
 * ref: https://fossies.org/linux/tensorflow/tensorflow/cc/framework/ops.h
 */

#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>
#include <atomic>

#include "abyss_export.h"
#include "types.h"
// #include "core/array.h"
// #include "core/visitor.h"

namespace abyss {

namespace core {
class Array;
class VisitorBase;
}  // namespace core

/**
 * @brief A type erased multi-dimensional array.
 */
class ABYSS_EXPORT Tensor {
 public:
  // template <typename T>
  // class Initializer;

  Tensor() = default;

  /**
   * @brief Construct Tensor from a managed array
   *
   * Performs type erasure on a managed array.
   * Takes a optional parameter shape to specify the Tensor shape.
   *
   * @tparam T the true type of the underlying array.
   * @param[in] data shared pointer to a ArrayImpl<T> object.
   * @param[in] shape shape of the Tensor.
   */
  // template <typename T>
  // Tensor(std::shared_ptr<core::ArrayImpl<T>> data,
  //        std::vector<int> shape = {0});

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
   *
   * The reason for defining one constructor per type is to prevent the use of a
   * template. This will ensure that oure core functionality is not mixed with
   * our API public headers.
   */
  Tensor(bool scalar);
  /**
   * @brief Construct Tensor from a scalar.
   */
  Tensor(uint8_t scalar);
  /**
   * @brief Construct Tensor from a scalar.
   */
  Tensor(int32_t scalar);
  /**
   * @brief Construct Tensor from a scalar.
   */
  Tensor(double scalar);
  /**
   * @brief Construct Tensor from a scalar.
   */
  Tensor(std::complex<double> scalar);

  // all members are copyable and movable, set to default
  Tensor(const Tensor& other) = default;
  Tensor(Tensor&& other) = default;

  Tensor& operator=(Tensor copy) {
    swap(*this, copy);

    return *this;
  }

  virtual ~Tensor() {
    // std::cout << "Tensor destruct" << std::endl;
  }

  friend void swap(Tensor& a, Tensor& b) noexcept {
    using std::swap;

    swap(a.dtype_, b.dtype_);
    swap(a.shape_, b.shape_);
    swap(a.strides_, b.strides_);
    swap(a.data_, b.data_);
  }

  // friend class core::VisitorBase;

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

  const ScalarType& dtype() const;
  const std::vector<int>& shape() const;
  const std::vector<int>& strides() const;
  core::Array* data() const;

  size_t size() const;
  size_t nbytes() const;
  size_t ndims() const;

  int shape(int index) const;
  Tensor reshape(std::vector<int> new_shape);

  /**
   * interaction with static types
   */
  operator bool();
  operator uint8_t();
  operator int32_t();
  operator double();

 protected:
  ScalarType dtype_ = kNone;
  std::vector<int> shape_;
  std::vector<int> strides_;
  size_t offset_ = 0;
  std::shared_ptr<core::Array> data_;
  // other properties for back-propagation
  // bool is_contiguous_ = true;
  // bool is_leaf = true;

  // Tensor* grad_;
  // core::VisitorBase* grad_fn_;
};

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
 * Tensor implementations
 */
// Tensor Tensor::empty(std::vector<int> shape, ScalarType dtype) {
//   Tensor out;
//   out.dtype_ = dtype;
//   out.shape_ = shape;
//   out.strides_ = core::shape2strides(shape);
//   out.data_ = dtype->create(shape);

//   return out;
// }

// template <typename T>
// Tensor::Tensor(T scalar) : dtype_{stypeof<T>()}, shape_{1}, strides_{1} {
//   data_ = std::make_shared<core::ArrayImpl<T>>(1, scalar);
// }
// template <typename T>
// Tensor::Tensor(T scalar) : dtype_{stypeof(scalar)}, shape_{1}, strides_{1} {
//   data_ = std::make_shared<core::ArrayImpl<T>>(1, scalar);
// }

// template <typename T>
// Tensor::Tensor(std::shared_ptr<core::ArrayImpl<T>> data, std::vector<int>
// shape)
//     : dtype_{stypeof<T>()}, data_{data} {
//   if (shape[0] == 0)
//     shape_ = {static_cast<int>(data->size())};
//   else
//     shape_ = shape;

//   strides_ = core::shape2strides(shape_);
// }

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

#endif  // ABYSS_TENSOR_H
