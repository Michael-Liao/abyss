#ifndef ABYSS_SCALARTYPE_H
#define ABYSS_SCALARTYPE_H

#include <type_traits>
#include <typeindex>
#include <ostream>

#include "abyss_export.h"
#include "core/dtype.h"
#include "core/traits.h"

namespace abyss {

class ABYSS_EXPORT ScalarType : public core::Dispatchable {
 public:
  template <typename T>
  ScalarType(T* dtype) : data_{dtype} {
    static_assert(std::is_base_of<core::DTypeBase, T>::value,
                  "should be a child of DTypeBase");
  }

  ScalarType(const ScalarType&) = default;

  ScalarType& operator=(ScalarType copy);

  std::type_index id() const { return data_->id(); }
  size_t itemsize() const { return data_->itemsize(); }

  bool operator==(const ScalarType& other) const;
  bool operator!=(const ScalarType& other) const;

 protected:
  core::DTypeBase* data_;

  core::DTypeBase* data() const override { return data_; }
  // void desc() { /* dummy */
  // }
};

// ABYSS_EXPORT extern const ScalarType kInt32;
ABYSS_EXPORT extern const ScalarType kNone;
ABYSS_EXPORT extern const ScalarType kBool;
ABYSS_EXPORT extern const ScalarType kUint8;
ABYSS_EXPORT extern const ScalarType kInt32;
ABYSS_EXPORT extern const ScalarType kFloat64;
ABYSS_EXPORT extern const ScalarType kComplex128;


template <typename T,
          std::enable_if_t<std::is_same<T, bool>::value, int> = 1>
ScalarType stypeof(T value = 0) {
  return kBool;
}

template <typename T,
          std::enable_if_t<std::is_same<T, uint8_t>::value, int> = 2>
ScalarType stypeof(T value = 0) {
  return kUint8;
}

template <typename T,
          std::enable_if_t<std::is_same<T, int32_t>::value, int> = 3>
ScalarType stypeof(T value = 0) {
  return kInt32;
}

template <typename T, std::enable_if_t<std::is_same<T, double>::value, int> = 4>
ScalarType stypeof(T value = 0) {
  return kFloat64;
}

template <
    typename T,
    std::enable_if_t<std::is_same<T, std::complex<double>>::value, int> = 5>
ScalarType stypeof(T value = 0) {
  return kComplex128;
}

template <typename T, std::enable_if_t<std::is_same<T, void>::value, int> = 0>
ScalarType stypeof(T value = 0) {
  return kNone;
}


}  // namespace abyss
#endif