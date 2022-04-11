#ifndef ABYSS_CORE_ITERATOR_H
#define ABYSS_CORE_ITERATOR_H

#include <iterator>

namespace abyss::core {

/**
 * @brief n-dimensional iterator.
 *
 * Specifies strides as an optional parameter to stride through different
 * dimensions. It matches all specs of random access iterator.
 */
template <typename T>
class NDIterator {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using iterator_category = std::random_access_iterator_tag;

  NDIterator() = default;
  NDIterator(pointer ptr, int stride = 1) : ptr_{ptr}, stride_{stride} {}
  // NDIterator(pointer ptr, int steps, int stride = 1) : ptr_{ptr}, steps_{steps}, stride_{stride} {}
  NDIterator(const NDIterator&) = default;
  NDIterator(NDIterator&&) = default;

  NDIterator& operator=(const NDIterator&) = default;
  NDIterator& operator=(NDIterator&&) = default;

  NDIterator& operator+=(int n) {
    ptr_ += n * stride_;
    return *this;
  }
  NDIterator& operator-=(int n) { ptr_ -= n * stride_; }

  ~NDIterator() = default;

  /**
   * @brief set stride of iterator.
   *
   * Changing strides at runtime is useful especially when printing?
   */
  void set_stride(int new_stride) { stride_ = new_stride; }

  NDIterator operator+(int n) {
    pointer ptr = ptr_ + (n * stride_);
    return NDIterator(ptr, stride_);
  }

  NDIterator operator-(int n) {
    pointer ptr = ptr_ - n * stride_;
    return NDIterator(ptr, stride_);
  }

  reference operator*() { return *ptr_; }
  pointer operator->() { return ptr_; }

  NDIterator& operator++() {
    // ++it
    ptr_ += stride_;
    return *this;
  }
  NDIterator& operator++(int) {
    // it++
    ptr_ += stride_;
    return *this;
  }
  NDIterator& operator--() {
    // --it
    ptr_ -= stride_;
    return *this;
  }
  NDIterator& operator--(int) {
    // it--
    ptr_ -= stride_;
    return *this;
  }

  difference_type operator-(const NDIterator& other) { return ptr_ - other.ptr_; }

  reference operator[](int offset) { return *(ptr_ + offset * stride_); }

  bool operator==(const NDIterator& other) const { return ptr_ == other.ptr_; }
  bool operator!=(const NDIterator& other) const { return !(*this == other); }
  bool operator<(const NDIterator& other) const { return ptr_ < other.ptr_; }
  bool operator>(const NDIterator& other) const { return ptr_ > other.ptr_; }
  bool operator>=(const NDIterator& other) const { return !(*this < other); }
  bool operator<=(const NDIterator& other) const { return !(*this > other); }

 private:
  pointer ptr_ = nullptr;
  // int steps_ = 1;
  int stride_ = 1;
};

/**
 * Extra operators
 */
// template <typename T>
// NDIterator<T> operator+(NDIterator<T> nditer, int n) {
//   auto ptr = nditer.ptr_ + n;
//   return NDIterator<T>(ptr, nditer.stride_);
// }
template <typename T>
NDIterator<T> operator+(int n, NDIterator<T> nditer) {
  return nditer + n;
}

// template <typename T>
// bool operator==(const NDIterator<T>& a, const NDIterator<T>& b) { return
// a.ptr_ == b.ptr_; } template <typename T> bool operator!=(const
// NDIterator<T>& a, const NDIterator<T>& b) { return !(a == b); } template
// <typename T> bool operator<(const NDIterator<T>& a, const NDIterator<T>& b) {
// return a.ptr_ < b.ptr_; } template <typename T> bool operator>(const
// NDIterator<T>& a, const NDIterator<T>& b) { return a.ptr_ > b.ptr_; }
// template <typename T>
// bool operator>=(const NDIterator<T>& a, const NDIterator<T>& b) { return !(a
// < b); } template <typename T> bool operator<=(const NDIterator<T>& a, const
// NDIterator<T>& b) { return !(a > b); }

}  // namespace abyss::core

#endif