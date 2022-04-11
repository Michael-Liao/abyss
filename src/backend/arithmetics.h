#ifndef ABYSS_BACKEND_ARITHMETICS_H
#define ABYSS_BACKEND_ARITHMETICS_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "abyss_export.h"

namespace abyss::backend {

ABYSS_EXPORT void add(const int32_t* in1, const int32_t* in2, const size_t& n,
         int32_t* out) noexcept;
ABYSS_EXPORT void add(const double* in1, const int32_t* in2, const size_t& n,
         double* out) noexcept;
ABYSS_EXPORT void add(const int32_t* in1, const double* in2, const size_t& n,
         double* out) noexcept;
ABYSS_EXPORT void add(const double* in1, const double* in2, const size_t& n,
         double* out) noexcept;

ABYSS_EXPORT void sub(const int32_t* in1, const int32_t* in2, const size_t& n,
         int32_t* out) noexcept;
ABYSS_EXPORT void sub(const double* in1, const int32_t* in2, const size_t& n,
         double* out) noexcept;
ABYSS_EXPORT void sub(const int32_t* in1, const double* in2, const size_t& n,
         double* out) noexcept;
ABYSS_EXPORT void sub(const double* in1, const double* in2, const size_t& n,
         double* out) noexcept;

ABYSS_EXPORT void mult(const int32_t* in1, const int32_t* in2, const size_t& n,
          int32_t* out) noexcept;
ABYSS_EXPORT void mult(const int32_t* in1, const double* in2, const size_t& n,
          double* out) noexcept;
ABYSS_EXPORT void mult(const double* in1, const int* in2, const size_t& n,
          double* out) noexcept;
ABYSS_EXPORT void mult(const double* in1, const double* in2, const size_t& n,
          double* out) noexcept;

ABYSS_EXPORT void div(const int32_t* in1, const int32_t* in2, const size_t& n,
         int32_t* out) noexcept;
ABYSS_EXPORT void div(const int32_t* in1, const double* in2, const size_t& n,
         double* out) noexcept;
ABYSS_EXPORT void div(const double* in1, const int32_t* in2, const size_t& n,
         double* out) noexcept;
ABYSS_EXPORT void div(const double* in1, const double* in2, const size_t& n,
         double* out) noexcept;

/**
 * @brief experimental interface
 *
 * these functions uses axpy blas routines, don't know if they are faster.
 */
template <typename T1, typename T2>
void xadd(const T1* in1, const T2* in2, const size_t& n, std::common_type_t<T1, T2>* out) {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] + in2[i];
  }
}
// void xsub(const int* in1, const int* in2, const size_t& n, int* out)
// noexcept; void xsub(const double* in1, const int* in2, const size_t& n,
//          double* out) noexcept;
// void xsub(const int* in1, const double* in2, const size_t& n,
//          double* out) noexcept;
// void xsub(const double* in1, const double* in2, const size_t& n,
//          double* out) noexcept;
}  // namespace abyss::backend

#endif