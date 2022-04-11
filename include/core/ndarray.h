#ifndef ABYSS_CORE_NDARRAY_H
#define ABYSS_CORE_NDARRAY_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "allocator.h"
#include "iterator.h"
#include "visitor.h"

namespace abyss::core {

class NDArrayBase {
 public:
  NDArrayBase() = default;
  ~NDArrayBase() = default;

  virtual size_t nbytes() const = 0;
  virtual size_t size() const = 0;
  virtual std::vector<int> shape() const = 0;
  virtual std::vector<int> strides() const = 0;
  virtual void* ptr() const = 0;
};

template <typename T>
class NDArray : public NDArrayBase {
 public:
  using value_type = T;
  using size_type = std::vector<int>;
  using allocator_type = Allocator<T>;

  /**
   * @brief custom iterator class that returns slices
   */
  class Iterator;

  using iterator = typename NDArray<T>::Iterator;
  using const_iterator = const iterator;
  /// @todo custom reverse iterator
  /// our iterator is considered a "stashing iterator"
  /// so std::reverse_iterator adaptor doesn't work
  class ReverseIterator;
  using reverse_iterator = ReverseIterator;
  using const_reverse_iterator = const reverse_iterator;

  NDArray() = default;
  NDArray(T value);
  NDArray(size_type shape);
  NDArray(size_type shape, value_type value);
  
  template <typename InputIt>
  NDArray(size_type shape, InputIt first, InputIt last);

  NDArray(const NDArray& other);
  NDArray(NDArray&& other);

  /**
   * @brief convert from ndarray of other types
   *
   * This is required because different types have different memory layouts.
   * Casting of data is required.
   */
  template <typename U>
  NDArray(const NDArray<U>& other);

  ~NDArray();

  NDArray& operator=(NDArray copy);
  void swap(NDArray& other);

  /**
   * getters
   */

  size_t nbytes() const override;
  size_t size() const override;
  std::vector<int> shape() const override;
  std::vector<int> strides() const override;
  void* ptr() const override;

  T* data() const;
  bool is_view() const;

  /**
   * element access
   */

  T at(int index) const;
  T& at(int index);
  T operator[](int index) const noexcept;
  T& operator[](int index) noexcept;

  /**
   * Iterators?
   */
  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
  reverse_iterator rbegin();
  reverse_iterator rend();
  const_reverse_iterator rbegin() const;
  const_reverse_iterator rend() const;

  /**
   * @brief Broadcast to a new shape
   */
  NDArray broadcast_to(std::vector<int> shape);

 private:
  std::vector<int> shape_;
  std::vector<int> strides_;
  size_t size_ = 0;
  allocator_type allocator_;
  T* data_ = nullptr;

  // view does not manage data
  bool is_view_ = false;

  static size_t shape2size(const std::vector<int>& shape);
  static std::vector<int> shape2strides(const std::vector<int>& shape);
};

template <typename T>
class NDArray<T>::Iterator {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = NDArray<T>;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::random_access_iterator_tag;
  // using iterator_category = std::forward_iterator_tag;

  Iterator() = default;
  Iterator(pointer self, T* ptr);

  Iterator(const Iterator& other) = default;
  Iterator(Iterator&& other) = default;

  Iterator& operator=(const Iterator& other) = default;
  Iterator& operator=(Iterator&& other) = default;

  ~Iterator() = default;

  void swap(Iterator& other);

  // dereferencing the pointer results in a view
  reference operator*();
  pointer operator->();

  Iterator& operator++();
  Iterator& operator++(int discard);
  Iterator& operator--();
  Iterator& operator--(int discard);

  Iterator operator+(int n);
  Iterator operator-(int n);

  difference_type operator-(const Iterator& other);

  reference operator[](int index);

  bool operator==(const Iterator& other) const;
  bool operator!=(const Iterator& other) const;
  bool operator>(const Iterator& other) const;
  bool operator<(const Iterator& other) const;
  bool operator>=(const Iterator& other) const;
  bool operator<=(const Iterator& other) const;

 private:
  int stride_ = 1; // stride for stepping
  value_type view_;
  pointer self_;
};

template <typename T>
class NDArray<T>::ReverseIterator {
  public:
  using difference_type = std::ptrdiff_t;
  using value_type = NDArray<T>;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::random_access_iterator_tag;
  
  ReverseIterator() = default;
  ReverseIterator(pointer self, T* ptr);

  ReverseIterator(const ReverseIterator& other) = default;
  ReverseIterator(ReverseIterator&& other) = default;
  
  ReverseIterator& operator=(const ReverseIterator& other) = default;
  ReverseIterator& operator=(ReverseIterator&& other) = default;

  void swap(ReverseIterator& other);

  reference operator*();
  pointer operator->();

  ReverseIterator& operator++();
  ReverseIterator& operator++(int discard);
  ReverseIterator& operator--();
  ReverseIterator& operator--(int discard);

  ReverseIterator operator+(int n);
  ReverseIterator operator-(int n);

  difference_type operator-(const ReverseIterator& other);

  reference operator[](int index);

  bool operator==(const ReverseIterator& other) const;
  bool operator!=(const ReverseIterator& other) const;
  bool operator>(const ReverseIterator& other) const;
  bool operator<(const ReverseIterator& other) const;
  bool operator>=(const ReverseIterator& other) const;
  bool operator<=(const ReverseIterator& other) const;

  private:
  int stride_ = 1;
  T* ptr_;
  value_type view_;
  pointer self_;
};

/**
 * Implementations of NDArray
 */

template <typename T>
NDArray<T>::NDArray(T value) : shape_{1}, strides_{1}, size_{1}, is_view_{false} {
  data_ = allocator_.allocate(size_);
  data_[0] = value;
}

template <typename T>
NDArray<T>::NDArray(std::vector<int> shape)
    : shape_{shape}, strides_{shape2strides(shape)}, size_{shape2size(shape)} {
  data_ = allocator_.allocate(size_);
}

template <typename T>
NDArray<T>::NDArray(std::vector<int> shape, T value) : NDArray{shape} {
  std::fill_n(data_, size_, value);
}


template <typename T>
template <typename InputIt>
NDArray<T>::NDArray(size_type shape, InputIt first, InputIt last) :
shape_{shape}, strides_{shape2strides(shape)}, size_{shape2size(shape)} {
  if (size_ != last - first) {
    throw std::runtime_error("array length does not match NDArray element size");
  }
  data_ = allocator_.allocate(size_);
  std::copy(first, last, data_);
}

template <typename T>
NDArray<T>::NDArray(const NDArray<T>& other)
    : shape_{other.shape_},
      strides_{other.strides_},
      size_{other.size_},
      is_view_{other.is_view_} {
  data_ = allocator_.allocate(size_);
  /// slices might not be contiguous, we need a better strategy
  std::copy_n(other.data_, size_, data_);
}

template <typename T>
NDArray<T>::NDArray(NDArray<T>&& other)
    : shape_{other.shape_},
      strides_{other.strides_},
      size_{other.size_},
      is_view_{other.is_view_} {
  data_ = other.data_;

  other.data_ = nullptr;
}

template <typename T>
template <typename U>
NDArray<T>::NDArray(const NDArray<U>& other)
    : shape_{other.shape()},
      strides_{other.strides()},
      size_{other.size()},
      is_view_{other.is_view_} {
  data_ = allocator_.allocate(size_);

  std::copy_n(other.data(), size_, data_);
}

template <typename T>
NDArray<T>::~NDArray() {
  if (!is_view_) {
    allocator_.deallocate(data_, size_);
  }
}

template <typename T>
void NDArray<T>::swap(NDArray& other) {
  using std::swap;

  swap(shape_, other.shape_);
  swap(strides_, other.strides_);
  swap(size_, other.size_);
  swap(allocator_, other.allocator_);
  swap(data_, other.data_);
  swap(is_view_, other.is_view_);
}

template <typename T>
NDArray<T>& NDArray<T>::operator=(NDArray<T> copy) {
  swap(copy);

  return *this;
}

template <typename T>
size_t NDArray<T>::nbytes() const {
  return sizeof(T) * size_;
}

template <typename T>
size_t NDArray<T>::size() const {
  return size_;
}
template <typename T>
std::vector<int> NDArray<T>::shape() const {
  return shape_;
}
template <typename T>
std::vector<int> NDArray<T>::strides() const {
  return strides_;
}
template <typename T>
void* NDArray<T>::ptr() const {
  return (void*)data_;
}

template <typename T>
T* NDArray<T>::data() const {
  return data_;
}

template <typename T>
bool NDArray<T>::is_view() const { return is_view_; }

template <typename T>
T NDArray<T>::at(int index) const {
  if (index >= size_) {
    throw std::out_of_range("NDArray index out of range");
  }

  return data_[index];
}
template <typename T>
T& NDArray<T>::at(int index) {
  if (index >= size_) {
    throw std::out_of_range("NDArray index out of range");
  }

  return data_[index];
}

template <typename T>
T NDArray<T>::operator[](int index) const noexcept {
  return data_[index];
}
template <typename T>
T& NDArray<T>::operator[](int index) noexcept {
  return data_[index];
}

template <typename T>
typename NDArray<T>::iterator NDArray<T>::begin() {
  return iterator(this, data_);
}

template <typename T>
typename NDArray<T>::iterator NDArray<T>::end() {
  return iterator(this, data_ + size_);
}

template <typename T>
typename NDArray<T>::const_iterator NDArray<T>::begin() const {
  return const_iterator(this, data_);
}

template <typename T>
typename NDArray<T>::const_iterator NDArray<T>::end() const {
  return const_iterator(this, data_ + size_);
}

  
template <typename T>
typename NDArray<T>::reverse_iterator NDArray<T>::rbegin() {
  return reverse_iterator(this, data_ + size_);
}
  
template <typename T>
typename NDArray<T>::reverse_iterator NDArray<T>::rend() {
  return reverse_iterator(this, data_);
}
  
template <typename T>
typename NDArray<T>::const_reverse_iterator NDArray<T>::rbegin() const {
  return const_reverse_iterator(this, data_ + size_);
}
  
template <typename T>
typename NDArray<T>::const_reverse_iterator NDArray<T>::rend() const {
  return const_reverse_iterator(this, data_);
}

template <typename T>
size_t NDArray<T>::shape2size(const std::vector<int>& shape) {
  auto length =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

  return static_cast<size_t>(length);
}

template <typename T>
std::vector<int> NDArray<T>::shape2strides(const std::vector<int>& shape) {
  std::vector<int> strides(shape.size(), 1);

  for (int i = shape.size() - 1; i >= 0; i--) {
    strides[i] = std::accumulate(shape.begin() + i + 1, shape.end(), 1,
                                 std::multiplies<int>());
  }

  return strides;
}

/**
 * NDArray::Iterator Implementation
 */

template <typename T>
NDArray<T>::Iterator::Iterator(NDArray<T>* self, T* ptr) : self_{self} {
  stride_ = self_->strides_[0];

  if (self_->shape_.size() == 1) {
    // is vector
    view_.shape_ = {1};
    view_.strides_ = {1};
    view_.size_ = 1;
  } else {
    // is n-dimensional
    for (int i = self_->shape_.size() - 1; i > 0; i--) {
      view_.shape_.emplace(view_.shape_.begin(), self_->shape_[i]);
      view_.strides_.emplace(view_.strides_.begin(), self_->strides_[i]);
    }

    view_.size_ = NDArray<T>::shape2size(view_.shape_);
  }
  view_.data_ = ptr;
  view_.is_view_ = true;
}

template <typename T>
void NDArray<T>::Iterator::swap(Iterator& other) {
  using std::swap;
  
  swap(stride_, other.stride_);
  swap(view_, other.view_);
  swap(self_, other.self_);
}

template <typename T>
NDArray<T>& NDArray<T>::Iterator::operator*() {
  return view_;
}
template <typename T>
NDArray<T>* NDArray<T>::Iterator::operator->() {
  return &view_;
}

template <typename T>
typename NDArray<T>::Iterator& NDArray<T>::Iterator::operator++() {
  // ++it
  view_.data_ += stride_;
  return *this;
}
template <typename T>
typename NDArray<T>::Iterator& NDArray<T>::Iterator::operator++(int discard) {
  // it++
  return ++(*this);
}
template <typename T>
typename NDArray<T>::Iterator& NDArray<T>::Iterator::operator--() {
  view_.data_ -=stride_;
  return *this;
}
template <typename T>
typename NDArray<T>::Iterator& NDArray<T>::Iterator::operator--(int discard) {
  return --(*this);
}

template <typename T>
typename NDArray<T>::Iterator NDArray<T>::Iterator::operator+(int n) {
  return Iterator(self_, view_.data_ + n * stride_);
}

template <typename T>
typename NDArray<T>::Iterator NDArray<T>::Iterator::operator-(int n) {
  return Iterator(self_, view_.data_ - n * stride_);
}

template <typename T>
std::ptrdiff_t NDArray<T>::Iterator::operator-(const Iterator& other) {
  int dist = (view_.data_ - other.view_.data_) / stride_;
  return dist;
}

template <typename T>
NDArray<T>& NDArray<T>::Iterator::operator[](int index) {
  view_.data_ = self_->data_ + index * stride_;

  return view_;
}

template <typename T>
bool NDArray<T>::Iterator::operator==(const Iterator& other) const {
  return view_.data_ == other.view_.data_;
}
template <typename T>
bool NDArray<T>::Iterator::operator!=(const Iterator& other) const {
  return view_.data_ != other.view_.data_;
}
template <typename T>
bool NDArray<T>::Iterator::operator>(const Iterator& other) const {
  return view_.data_ > other.view_.data_;
}
template <typename T>
bool NDArray<T>::Iterator::operator<(const Iterator& other) const {
  return view_.data_ < other.view_.data_;
}
template <typename T>
bool NDArray<T>::Iterator::operator>=(const Iterator& other) const {
  return view_.data_ >= other.view_.data_;
}
template <typename T>
bool NDArray<T>::Iterator::operator<=(const Iterator& other) const {
  return view_.data_ <= other.view_.data_;
}

/**
 * NDArray::ReverseIterator Implementation
 */
template <typename T>
NDArray<T>::ReverseIterator::ReverseIterator(NDArray<T>* self, T* ptr) : self_{self}, ptr_{ptr} {
  stride_ = self_->strides_[0];

  if (self_->shape_.size() == 1) {
    // is vector
    view_.shape_ = {1};
    view_.strides_ = {1};
    view_.size_ = 1;
  } else {
    // is n-dimensional
    for (int i = self_->shape_.size() - 1; i > 0; i--) {
      view_.shape_.emplace(view_.shape_.begin(), self_->shape_[i]);
      view_.strides_.emplace(view_.strides_.begin(), self_->strides_[i]);
    }

    view_.size_ = NDArray<T>::shape2size(view_.shape_);
  }
  view_.data_ = ptr;
  view_.is_view_ = true;
}

template <typename T>
void NDArray<T>::ReverseIterator::swap(ReverseIterator& other) {
  using std::swap;
  
  swap(stride_, other.stride_);
  swap(ptr_, other.ptr_);
  swap(view_, other.view_);
  swap(self_, other.self_);
}

template <typename T>
NDArray<T>& NDArray<T>::ReverseIterator::operator*() {
  view_.data_ = ptr_ - stride_;
  return view_;
}
template <typename T>
NDArray<T>* NDArray<T>::ReverseIterator::operator->() {
  view_.data_ = ptr_ - stride_;
  return &view_;
}

template <typename T>
typename NDArray<T>::ReverseIterator& NDArray<T>::ReverseIterator::operator++() {
  // ++it
  ptr_ -= stride_;
  return *this;
}
template <typename T>
typename NDArray<T>::ReverseIterator& NDArray<T>::ReverseIterator::operator++(int discard) {
  // it++
  return ++(*this);
}
template <typename T>
typename NDArray<T>::ReverseIterator& NDArray<T>::ReverseIterator::operator--() {
  ptr_ +=stride_;
  return *this;
}
template <typename T>
typename NDArray<T>::ReverseIterator& NDArray<T>::ReverseIterator::operator--(int discard) {
  return --(*this);
}

template <typename T>
typename NDArray<T>::ReverseIterator NDArray<T>::ReverseIterator::operator+(int n) {
  return ReverseIterator(self_, ptr_ - n * stride_);
}

template <typename T>
typename NDArray<T>::ReverseIterator NDArray<T>::ReverseIterator::operator-(int n) {
  return ReverseIterator(self_, ptr_ + n * stride_);
}

template <typename T>
std::ptrdiff_t NDArray<T>::ReverseIterator::operator-(const ReverseIterator& other) {
  int dist = (other.ptr_ - ptr_ ) / stride_;
  return dist;
}

template <typename T>
NDArray<T>& NDArray<T>::ReverseIterator::operator[](int index) {
  view_.data_ = self_->data_ - index * stride_ - 1;
  return view_;
}

template <typename T>
bool NDArray<T>::ReverseIterator::operator==(const ReverseIterator& other) const {
  return ptr_ == other.ptr_;
}
template <typename T>
bool NDArray<T>::ReverseIterator::operator!=(const ReverseIterator& other) const {
  return ptr_ != other.ptr_;
}
template <typename T>
bool NDArray<T>::ReverseIterator::operator>(const ReverseIterator& other) const {
  return ptr_ > other.ptr_;
}
template <typename T>
bool NDArray<T>::ReverseIterator::operator<(const ReverseIterator& other) const {
  return ptr_ < other.ptr_;
}
template <typename T>
bool NDArray<T>::ReverseIterator::operator>=(const ReverseIterator& other) const {
  return ptr_ >= other.ptr_;
}
template <typename T>
bool NDArray<T>::ReverseIterator::operator<=(const ReverseIterator& other) const {
  return ptr_ <= other.ptr_;
}

}  // namespace abyss::core

#endif