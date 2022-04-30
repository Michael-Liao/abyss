#include "types.h"

// #include "scalartype.h"
// #include "core/dtype.h"

namespace abyss {

namespace detail {
static core::DTypeImpl<void> kNone_;
static core::DTypeImpl<bool> kBool_;
static core::DTypeImpl<uint8_t> kUint8_;
static core::DTypeImpl<int32_t> kInt32_;
static core::DTypeImpl<double> kFloat64_;
static core::DTypeImpl<std::complex<double>> kComplex128_;
}  // namespace detail

const ScalarType kNone(&detail::kNone_);
const ScalarType kBool = &detail::kBool_;
const ScalarType kUint8 = &detail::kUint8_;
// const ScalarType kInt32 = &detail::kInt32_;
const ScalarType kFloat64 = &detail::kFloat64_;
const ScalarType kComplex128 = &detail::kComplex128_;


}  // namespace abyss