#ifndef ABYSS_BACKEND_ARR_MATH_H
#define ABYSS_BACKEND_ARR_MATH_H

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "abyss_export.h"

namespace abyss::backend {
ABYSS_EXPORT void exp(const uint8_t* in_data, const size_t* ids, const size_t n,
                      uint8_t* out_data);
ABYSS_EXPORT void exp(const int32_t* in_data, const size_t* ids, const size_t n,
                      int32_t* out_data);
ABYSS_EXPORT void exp(const double* in_data, const size_t* ids, const size_t n,
                      double* out_data);


ABYSS_EXPORT void log(const uint8_t* in_data, const size_t* ids, const size_t n,
                      uint8_t* out_data);
ABYSS_EXPORT void log(const int32_t* in_data, const size_t* ids, const size_t n,
                      int32_t* out_data);
ABYSS_EXPORT void log(const double* in_data, const size_t* ids, const size_t n,
                      double* out_data);

// ABYSS_EXPORT void pow(const int32_t* base, const size_t* base_id,
//                       const int32_t* exp, const size_t* exp_id,
//                       const size_t n, int32_t* out_data);
// ABYSS_EXPORT void pow(const int32_t* base, const size_t* base_id,
//                       const double* exp, const size_t* exp_id,
//                       const size_t n, double* out_data);
// ABYSS_EXPORT void pow(const double* base, const size_t* base_id,
//                       const int32_t* exp, const size_t* exp_id,
//                       const size_t n, double* out_data);
// ABYSS_EXPORT void pow(const double* base, const size_t* base_id,
//                       const double* exp, const size_t* exp_id,
//                       const size_t n, double* out_data);
}  // namespace abyss::backend

#endif