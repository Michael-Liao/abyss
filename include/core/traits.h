#ifndef ABYSS_TRAITS_H
#define ABYSS_TRAITS_H

/**
 * @file This file defines type traits to interact with the native type system.
 */

#include <complex>
#include <functional>
#include <unordered_map>

// #include "core/dtype.h"
#include "visitor.h"
#include "utility.h"
// #include "scalartype.h"
// #include "types.h"

namespace abyss::core {

template <typename T>
struct is_supported_dtype
    : std::integral_constant<bool,
                             std::is_arithmetic<T>::value ||
                                 std::is_same<T, std::complex<float>>::value ||
                                 std::is_same<T, std::complex<double>>::value> {
};

// template <typename T,
//           std::enable_if_t<std::is_same<T, bool>::value, int> = 1>
// ScalarType stypeof(T value = 0) {
//   return kBool;
// }

// template <typename T,
//           std::enable_if_t<std::is_same<T, uint8_t>::value, int> = 2>
// ScalarType stypeof(T value = 0) {
//   return kUint8;
// }

// template <typename T,
//           std::enable_if_t<std::is_same<T, int32_t>::value, int> = 3>
// ScalarType stypeof(T value = 0) {
//   return kInt32;
// }

// template <typename T, std::enable_if_t<std::is_same<T, double>::value, int> = 4>
// ScalarType stypeof(T value = 0) {
//   return kFloat64;
// }

// template <
//     typename T,
//     std::enable_if_t<std::is_same<T, std::complex<double>>::value, int> = 5>
// ScalarType stypeof(T value = 0) {
//   return kComplex128;
// }

// template <typename T, std::enable_if_t<std::is_same<T, void>::value, int> = 0>
// ScalarType stypeof(T value = 0) {
//   return kNone;
// }

class Array;

template <typename T>
class ArrayImpl;

// class VisitorBase;

/**
 * @brief Visitable type trait
 *
 * Any type that supports visit should inherit from this type and implement all
 * `accept` functions.
 */
class Visitable {
 public:
  // accepts unary functions
  virtual void accept(VisitorBase*) = 0;

  // accepts binary functions (triple dispatch)
  virtual void accept(VisitorBase*, Visitable*) = 0;
  virtual void accept(VisitorBase*, ArrayImpl<bool>*) = 0;
  virtual void accept(VisitorBase*, ArrayImpl<uint8_t>*) = 0;
  virtual void accept(VisitorBase*, ArrayImpl<int32_t>*) = 0;
  virtual void accept(VisitorBase*, ArrayImpl<double>*) = 0;
};

/**
 * @brief a type trait to mark things that can be decorated by the `Dispatcher`.
 */
// class Dispatchable {
//   public:
//   virtual Visitable* data() const = 0;
//   virtual ArrayDesc desc() const { return ArrayDesc(); }
// };

// namespace detail {
  
// template <bool...>
// class bool_pack {};

// template <typename... Ts>
// using conjunction =
//     std::is_same<bool_pack<true, Ts::value...>, bool_pack<Ts::value..., true>>;

// } // namespace detail


}  // namespace abyss

#endif