#ifndef ABYSS_CORE_ARRAY_H
#define ABYSS_CORE_ARRAY_H

/**
 * The proper Array interface
 */
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <iostream>
#include <numeric>
#include <vector>

#include "allocator.h"
#include "iterator.h"
#include "visitor.h"
#include "traits.h"
// #include "types.h"

namespace abyss::core {


struct Array : Visitable {
  virtual ~Array() = default;

  virtual size_t size() const = 0;
  /**
   * @brief set all data back to 0
   */
  virtual void zero() = 0;
};

template <typename T>
class ArrayImpl : public Array {
 public:
  using allocator_type = Allocator<T>;
  using iterator = StridedIterator<T>;
  using const_iterator = const StridedIterator<T>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  static ArrayImpl<T> from_range(T start, T stop, T step) {
    size_t size = (stop - start + step - 1) / step;
    ArrayImpl<T> out(size);

    for (size_t i = 0; i < size; i++) {
      out.data_[i] = start + i * step;
    }

    return out;
  }
  static ArrayImpl<T> from_range(T stop) { return from_range(0, stop, 1); }

  ArrayImpl() = default;

  // trivially copy/move constructible (shallow copy)
  // ArrayImpl(const ArrayImpl& other) = default;
  // ArrayImpl(ArrayImpl&& other) = default;

  // deep copy/move constructors
  ArrayImpl(const ArrayImpl& other);
  ArrayImpl(ArrayImpl&& other);

  /// @todo copy construct from another type
  template <typename U>
  ArrayImpl(const ArrayImpl<U>& other);

  ArrayImpl(size_t size);
  ArrayImpl(size_t size, T value);

  ArrayImpl(std::vector<T> values);
  ArrayImpl(std::initializer_list<T> values);

  ArrayImpl& operator=(ArrayImpl copy);

  ~ArrayImpl();

  size_t size() const override { return size_; }
  void zero() override {
    std::fill_n(data_, size_, 0);
  }

  T* data() const { return data_; }

  T& at(int offset) {
    if (offset >= size_) throw std::out_of_range("Array index out of range.");
    return *(data_ + offset);
  }
  const T& at(int offset) const {
    if (offset >= size_) throw std::out_of_range("Array index out of range.");
    return *(data_ + offset);
  }
  T& operator[](int offset) noexcept { return *(data_ + offset); }
  const T& operator[](int offset) const noexcept { return *(data_ + offset); }

  iterator begin(int stride = 1) { return iterator(data_, stride); }
  iterator end() { return iterator(data_ + size_); }
  const_iterator begin(int stride = 1) const { return iterator(data_, stride); }
  const_iterator end() const { return iterator(data_ + size_); }
  reverse_iterator rbegin(int stride = 1) {
    return reverse_iterator(data_, stride);
  }
  reverse_iterator rend() { return reverse_iterator(data_ + size_); }
  const_reverse_iterator rbegin(int stride = 1) const {
    return reverse_iterator(data_, stride);
  }
  const_reverse_iterator rend() const {
    return reverse_iterator(data_ + size_);
  }

  NDIterator<T> nbegin(ArrayDesc desc) { return NDIterator<T>(data_, desc); }
  NDIterator<T> nend(ArrayDesc desc) { 
    size_t real_size = shape2size(desc.shape);
    return NDIterator<T>(data_, desc, real_size);
  }

  void swap(ArrayImpl& other) noexcept;

  void accept(VisitorBase*) override;

  void accept(VisitorBase*, Visitable*) override;
  void accept(VisitorBase*, ArrayImpl<bool>*) override;
  void accept(VisitorBase*, ArrayImpl<uint8_t>*) override;
  void accept(VisitorBase*, ArrayImpl<int32_t>*) override;
  void accept(VisitorBase*, ArrayImpl<double>*) override;

  //  protected:

 private:
  size_t size_ = 0;
  allocator_type allocator_;
  T* data_ = nullptr;
};

/**
 * helper functions
 */

// inline size_t shape2size(const std::vector<int>& shape) {
//   auto length =
//       std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

//   return static_cast<size_t>(length);
// }

// inline std::vector<int> shape2strides(const std::vector<int>& shape) {
//   std::vector<int> strides(shape.size(), 1);

//   for (int i = shape.size() - 1; i >= 0; i--) {
//     strides[i] = std::accumulate(shape.begin() + i + 1, shape.end(), 1,
//                                  std::multiplies<int>());
//   }

//   return strides;
// }

// inline std::vector<int> unravel_index(const int index,
//                                       const std::vector<int>& shape) {
//   std::vector<int> out(shape.size());

//   int quotient = index;
//   for (int i = shape.size() - 1; i >= 0; i--) {
//     out[i] = quotient % shape[i];
//     quotient = quotient / shape[i];
//   }

//   return out;
// }

// /**
//  * @brief Broadcast and copy the input sequence into an output
//  * 
//  * @tparam InputIt Input iterator of the input sequence.
//  * @tparam OutputIt Output iterator for thee destination sequence.
//  * @param first The start of the input sequence.
//  * @param last The end of the input sequence.
//  * @param shape The shape of the input.
//  * @param d_first The start of the output sequence.
//  * @param d_shape Target shape the broadcast to.
//  */
// template <typename InputIt, typename OutputIt>
// OutputIt broadcast_copy(InputIt first, InputIt last, const std::vector<int>& shape,
//                     OutputIt d_first, const std::vector<int>& d_shape) {
//   OutputIt d_next = d_first;
//   std::vector<int> n_broadcast = d_shape;

//   auto n_bc_it = n_broadcast.rbegin();
//   auto shp_it = shape.rbegin();
//   while (shp_it != shape.rend()) {
//     *n_bc_it++ /= *shp_it++;
//   }

//   using namespace std::placeholders; // for _1

//   auto neq_one = std::bind(std::not_equal_to<>(), _1, 1);
//   int first_bc_axis =
//       std::find_if(n_broadcast.rbegin(), n_broadcast.rbegin() + shape.size(),
//                    neq_one) -
//       n_broadcast.rbegin();

//   int stride = 1;
//   int n_copies = 1;
//   if (first_bc_axis == shape.size()) {
//     // broadcast not required
//     d_next = std::copy(first, last, d_first);

//     n_copies = std::accumulate(n_broadcast.rbegin() + shape.size(),
//                                n_broadcast.rend(), 1, std::multiplies<>());
//   } else {
//     // first broadcast refers to the original shape
//     stride = std::accumulate(shape.rbegin(), shape.rbegin() + first_bc_axis, 1,
//                              std::multiplies<>());

//     int bc_size = *(n_broadcast.rbegin() + first_bc_axis);
//     // first broadcast copies data from the original array
//     while (first < last) {
//       for (int i = 0; i < bc_size; i++) {
//         d_next = std::copy_n(first, stride, d_next);
//       }
//       first += stride;
//     }

//     n_copies = std::accumulate(n_broadcast.rbegin() + first_bc_axis + 1,
//                                n_broadcast.rend(), 1, std::multiplies<>());
//   }


//   stride = d_next - d_first;
//   for (size_t i = 0; i < n_copies - 1; i++) {
//     d_next = std::copy_n(d_first, stride, d_next);
//   }

//   return d_next;
// }

/**
 * Implementations
 */

template <typename T>
ArrayImpl<T>::ArrayImpl(const ArrayImpl& other) {
  if (size_ != other.size_) {
    // only reallocate when the size is different
    allocator_.deallocate(data_, size_);

    size_ = other.size_;
    allocator_ = other.allocator_;
    data_ = allocator_.allocate(size_);
  }
  std::copy_n(other.data_, other.size_, data_);
}

template <typename T>
ArrayImpl<T>::ArrayImpl(ArrayImpl&& other) {
  allocator_.deallocate(data_, size_);

  size_ = other.size_;
  allocator_ = other.allocator_;
  data_ = other.data_;

  other.data_ = nullptr;
}

template <typename T>
template <typename U>
ArrayImpl<T>::ArrayImpl(const ArrayImpl<U>& other) : size_{other.size()} {
  data_ = allocator_.allocate(size_);
  std::copy(other.begin(), other.end(), data_);
}

template <typename T>
ArrayImpl<T>::ArrayImpl(size_t size) : size_{size} {
  data_ = allocator_.allocate(size);
}

template <typename T>
ArrayImpl<T>::ArrayImpl(size_t size, T value) : size_{size} {
  data_ = allocator_.allocate(size);
  std::fill_n(data_, size_, value);
}

template <typename T>
ArrayImpl<T>::ArrayImpl(std::vector<T> values) : size_{values.size()} {
  data_ = allocator_.allocate(size_);
  std::copy(values.begin(), values.end(), data_);
}

template <typename T>
ArrayImpl<T>::ArrayImpl(std::initializer_list<T> values) : size_{values.size()} {
  data_ = allocator_.allocate(size_);
  std::copy(values.begin(), values.end(), data_);
}

// template <typename T>
// ArrayImpl<T>::ArrayImpl(ArrayImpl&& other) {}

template <typename T>
ArrayImpl<T>& ArrayImpl<T>::operator=(ArrayImpl copy) {
  swap(*this, copy);

  return *this;
}

template <typename T>
ArrayImpl<T>::~ArrayImpl() {
  allocator_.deallocate(data_, size_);
}

template <typename T>
void ArrayImpl<T>::accept(VisitorBase* vis) {
  auto visitor = dynamic_cast<UnaryVisitor<ArrayImpl<T>>*>(vis);
  visitor->visit(this);
}

template <typename T>
void ArrayImpl<T>::accept(VisitorBase* vis, Visitable* b) {
  // std::cout<<"ArrayImpl accept (meta) > ";
  b->accept(vis, this);
}

template <typename T>
void ArrayImpl<T>::accept(VisitorBase* vis, ArrayImpl<bool>* a) {
  auto visitor =
      dynamic_cast<BinaryVisitor<ArrayImpl<bool>, ArrayImpl<T>>*>(vis);
  visitor->visit(a, this);
}
template <typename T>
void ArrayImpl<T>::accept(VisitorBase* vis, ArrayImpl<uint8_t>* a) {
  auto visitor =
      dynamic_cast<BinaryVisitor<ArrayImpl<uint8_t>, ArrayImpl<T>>*>(vis);
  visitor->visit(a, this);
}
template <typename T>
void ArrayImpl<T>::accept(VisitorBase* vis, ArrayImpl<int32_t>* a) {
  // std::cout<<"ArrayImpl accept." << std::endl;
  auto visitor =
      dynamic_cast<BinaryVisitor<ArrayImpl<int32_t>, ArrayImpl<T>>*>(vis);
  visitor->visit(a, this);
}
template <typename T>
void ArrayImpl<T>::accept(VisitorBase* vis, ArrayImpl<double>* a) {
  auto visitor =
      dynamic_cast<BinaryVisitor<ArrayImpl<double>, ArrayImpl<T>>*>(vis);
  visitor->visit(a, this);
}

template <typename T>
void ArrayImpl<T>::swap(ArrayImpl<T>& other) noexcept {
  using std::swap;

  swap(size_, other.size_);
  swap(allocator_, other.allocator_);
  swap(data_, other.data_);
}

template <typename T>
void swap(ArrayImpl<T>& a, ArrayImpl<T>& b) noexcept {
  a.swap(b);
}

}  // namespace abyss::core

#endif  // ABYSS_CORE_ARRAY_H