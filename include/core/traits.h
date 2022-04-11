#ifndef ABYSS_TRAITS_H
#define ABYSS_TRAITS_H

/**
 * @file This file defines type traits to interact with the native type system.
 */

#include <complex>
#include <functional>
#include <unordered_map>

// #include "core/dtype.h"
#include "types.h"

namespace abyss {

template <typename T>
struct is_supported_dtype
    : std::integral_constant<bool,
                             std::is_arithmetic<T>::value ||
                                 std::is_same<T, std::complex<float>>::value ||
                                 std::is_same<T, std::complex<double>>::value> {
};

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

// inline ScalarType stypeof(uint8_t value) { return kUint8; }
// inline ScalarType stypeof(int32_t value) { return kInt32; }
// inline ScalarType stypeof(double value) { return kFloat64; }
// inline ScalarType stypeof(std::complex<double> value) { return kComplex128; }

}  // namespace abyss

#endif