#ifndef ABYSS_ALLOCATOR_H
#define ABYSS_ALLOCATOR_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <utility>

// #include "types.h"
// #include "buffer.h"

namespace abyss {

/**
 * @brief Allocator for the n-dimensional container
 * 
 * This allocator is currently a basic allocator.
 * In the future this will expand into a allocator that allocates device and host memory.
 * This class should match the name requirements of Allocator.
 */
template <typename T>
class Allocator {
 public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

  Allocator() = default;
  Allocator(const Allocator& other) = default;
  Allocator(Allocator&& other) = default;

  ~Allocator() = default;

  Allocator& operator=(const Allocator& other) = default;
  Allocator& operator=(Allocator&& other) = default;

  T* allocate(size_type n) {
    if (n > std::numeric_limits<size_type>::max() / sizeof(T))
      throw std::bad_array_new_length();

    T* ptr = static_cast<T*>(std::malloc(n * sizeof(T)));
    if (ptr) return ptr;

    throw std::bad_alloc();
  }
  void deallocate(T* ptr, size_type n) { std::free(ptr); }

//  private:
  // size_type size_ = 0;
  //  pointer data_ = nullptr;
};

template <typename T1, typename T2>
bool operator==(Allocator<T1> a, Allocator<T2> b) {
  return std::is_same<T1, T2>::value;
}

template <typename T1, typename T2>
bool operator!=(Allocator<T1> a, Allocator<T2> b) {
  return !(a == b);
}

}  // namespace abyss

#endif  // ABYSS_ALLOCATOR_H
