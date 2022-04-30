#ifndef ABYSS_TYPES_H
#define ABYSS_TYPES_H

/**
 * Supported type definitions.
 * 
 * Using the `DType` objects, all our types are visitable.
 * This allows conversion can array creation with a type erased `Tensor`.
 */

#include <complex>
#include <cstddef>
#include <cstdint>
#include <typeindex>
#include <memory>
#include <vector>
#include <typeinfo>

#include "abyss_export.h"
// #include "scalartype.h"
// #include "buffer.h"
// #include "core/array.h"
// #include "core/dtype.h"
// #include "traits.h"

namespace abyss {

// namespace core {
// class DTypeBase;
// }

// using ScalarType = core::DTypeBase*;
// class ScalarType {};

// extern DType* kNone;
// extern DType* kUint8;
// extern DType* kInt32;
// extern DType* kFloat64;
// extern DType* kComplex128;

ABYSS_EXPORT extern const ScalarType kNone;
ABYSS_EXPORT extern const ScalarType kBool;
ABYSS_EXPORT extern const ScalarType kUint8;
ABYSS_EXPORT extern const ScalarType kInt32;
ABYSS_EXPORT extern const ScalarType kFloat64;
ABYSS_EXPORT extern const ScalarType kComplex128;

// static const ScalarType kNone = ScalarType::from_native_type<void>();
// static const ScalarType kBool = ScalarType::from_native_type<bool>();
// static const ScalarType kInt8 = ScalarType::from_native_type<int8_t>();
// static const ScalarType kUint8 = ScalarType::from_native_type<uint8_t>();
// static const ScalarType kInt16 = ScalarType::from_native_type<int16_t>();
// static const ScalarType kUint16 = ScalarType::from_native_type<uint16_t>();
// static const ScalarType kInt32 = ScalarType::from_native_type<int32_t>();
// static const ScalarType kUint32 = ScalarType::from_native_type<uint32_t>();
// static const ScalarType kInt64 = ScalarType::from_native_type<int64_t>();
// static const ScalarType kUint64 = ScalarType::from_native_type<uint64_t>();
// static const ScalarType kFloat32 = ScalarType::from_native_type<float>();
// static const ScalarType kFloat64 = ScalarType::from_native_type<double>();
// static const ScalarType kComplex64 =
//     ScalarType::from_native_type<std::complex<float>>();
// static const ScalarType kComplex128 =
//     ScalarType::from_native_type<std::complex<double>>();

}  // namespace abyss

#endif
