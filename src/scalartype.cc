#include "scalartype.h"

#include <complex>

namespace abyss {

ScalarType& ScalarType::operator=(ScalarType copy) {
  std::swap(dyn_type_, copy.dyn_type_);

  return *this;
}
bool ScalarType::operator==(const ScalarType& other) const {
  return dyn_type_ == other.dyn_type_;
}
bool ScalarType::operator!=(const ScalarType& other) const {
  return dyn_type_ != other.dyn_type_;
}
namespace details {
static core::DTypeImpl<int32_t> kNone_;
static core::DTypeImpl<bool> kBool_;
static core::DTypeImpl<uint8_t> kUint8_;
static core::DTypeImpl<int32_t> kInt32_;
static core::DTypeImpl<double> kFloat64_;
static core::DTypeImpl<std::complex<double>> kComplex128_;
}

const ScalarType kNone(&details::kNone_);
const ScalarType kBool(&details::kBool_);
const ScalarType kUint8(&details::kUint8_);
const ScalarType kInt32(&details::kInt32_);
const ScalarType kFloat64(&details::kFloat64_);
const ScalarType kComplex128(&details::kComplex128_);

}