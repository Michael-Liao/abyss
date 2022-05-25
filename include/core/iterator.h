#ifndef ABYSS_CORE_ITERATOR_H
#define ABYSS_CORE_ITERATOR_H

#include <iterator>

#include "utility.h"

namespace abyss::core {

/**
 * @brief iterator with strides
 *
 * Specifies strides as an optional parameter to stride through different
 * dimensions. It matches all specs of random access iterator.
 */
template <typename T>
class StridedIterator {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using iterator_category = std::random_access_iterator_tag;

  StridedIterator() = default;
  StridedIterator(pointer ptr, int stride = 1) : ptr_{ptr}, stride_{stride} {}
  StridedIterator(const StridedIterator&) = default;
  StridedIterator(StridedIterator&&) = default;

  StridedIterator& operator=(const StridedIterator&) = default;
  StridedIterator& operator=(StridedIterator&&) = default;

  StridedIterator& operator+=(int n) {
    ptr_ += n * stride_;
    return *this;
  }
  StridedIterator& operator-=(int n) {
    ptr_ -= n * stride_;
    return *this;
  }

  ~StridedIterator() = default;

  StridedIterator operator+(int n) {
    pointer ptr = ptr_ + (n * stride_);
    return StridedIterator(ptr, stride_);
  }

  StridedIterator operator-(int n) {
    pointer ptr = ptr_ - n * stride_;
    return StridedIterator(ptr, stride_);
  }

  reference operator*() { return *ptr_; }
  pointer operator->() { return ptr_; }

  StridedIterator& operator++() {
    // ++it
    ptr_ += stride_;
    return *this;
  }
  StridedIterator& operator++(int) {
    // it++
    ptr_ += stride_;
    return *this;
  }
  StridedIterator& operator--() {
    // --it
    ptr_ -= stride_;
    return *this;
  }
  StridedIterator& operator--(int) {
    // it--
    ptr_ -= stride_;
    return *this;
  }

  difference_type operator-(const StridedIterator& other) {
    return ptr_ - other.ptr_;
  }

  reference operator[](int offset) { return *(ptr_ + offset * stride_); }

  bool operator==(const StridedIterator& other) const {
    return ptr_ == other.ptr_;
  }
  bool operator!=(const StridedIterator& other) const {
    return !(*this == other);
  }
  bool operator<(const StridedIterator& other) const {
    return ptr_ < other.ptr_;
  }
  bool operator>(const StridedIterator& other) const {
    return ptr_ > other.ptr_;
  }
  bool operator>=(const StridedIterator& other) const {
    return !(*this < other);
  }
  bool operator<=(const StridedIterator& other) const {
    return !(*this > other);
  }

 private:
  pointer ptr_ = nullptr;
  // int steps_ = 1;
  int stride_ = 1;
};

/**
 * Extra operators for StridedIterator
 */
template <typename T>
StridedIterator<T> operator+(int n, StridedIterator<T> nditer) {
  return nditer + n;
}

template <typename T>
class NDIterator {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using iterator_category = std::random_access_iterator_tag;

  NDIterator(pointer ptr, ArrayDesc desc, int index = 0) : ptr_{ptr}, index_{index}, desc_{desc} {}

  NDIterator(const NDIterator&) = default;

  NDIterator& operator+=(int n) {
    index_ += n;
    return *this;
  }
  NDIterator& operator-=(int n) {
    index_ -= n;
    return *this;
  }

  ~NDIterator() = default;

  reference operator*() {
    size_t offset = calc_offset(index_);
    return *(ptr_ + offset);
  }
  pointer operator->() {
    size_t offset = calc_offset(index_);
    return ptr_ + offset;
  }

  NDIterator& operator++() {
    // ++it
    *this += 1;
    return *this;
  }
  NDIterator& operator++(int) {
    // it++
    *this += 1;
    return *this;
  }
  NDIterator& operator--() {
    // --it
    *this -= 1;
    return *this;
  }
  NDIterator& operator--(int) {
    // it--
    *this -= 1;
    return *this;
  }

  NDIterator operator+(int n) {
    return NDIterator(ptr_, desc_, index_ + n);
  }

  NDIterator operator-(int n) {
    return NDIterator(ptr_, desc_, index_ - n);
  }

  difference_type operator-(const NDIterator& other) {
    return index_ - other.index_;
  }

  reference operator[](int shift) {
    size_t offset = calc_offset(index_ + shift);
    return *(ptr_ + offset);
  }

  bool operator==(const NDIterator& other) const { return ptr_ == other.ptr_; }
  bool operator!=(const NDIterator& other) const { return !(*this == other); }
  bool operator<(const NDIterator& other) const { return ptr_ < other.ptr_; }
  bool operator>(const NDIterator& other) const { return ptr_ > other.ptr_; }
  bool operator>=(const NDIterator& other) const { return !(*this < other); }
  bool operator<=(const NDIterator& other) const { return !(*this > other); }

 private:
  pointer ptr_ = nullptr;
  int index_ = 0;
  ArrayDesc desc_;

  size_t calc_offset(int index) {
    auto coords = unravel_index(index, desc_.shape);

    size_t offset = desc_.offset;
    for (size_t i = 0; i < coords.size(); i++) {
      offset += coords[i] * desc_.strides[i];
    }

    return offset;
  }
};

}  // namespace abyss::core

#endif