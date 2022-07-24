#ifndef ABYSS_BACKEND_ARITHMETICS_H
#define ABYSS_BACKEND_ARITHMETICS_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "abyss_export.h"

namespace abyss::backend {

// ABYSS_EXPORT void add(const int32_t* in1, const int32_t* in2, const size_t& n,
//                       int32_t* out) noexcept;
// ABYSS_EXPORT void add(const double* in1, const int32_t* in2, const size_t& n,
//                       double* out) noexcept;
// ABYSS_EXPORT void add(const int32_t* in1, const double* in2, const size_t& n,
//                       double* out) noexcept;
// ABYSS_EXPORT void add(const double* in1, const double* in2, const size_t& n,
//                       double* out) noexcept;

/**
 * @brief add function (new signature)
 *
 * making the backend function smarter gives us more flexibility
 * to support both broadcasting and slicing directly in low-level functions
 *
 * @param[in] in_data1 input data 1 (lhs)
 * @param[in] id1 input data 1 indices
 * @param[in] in_data2 input data 2 (rhs)
 * @param[in] id2 input data 2 indices
 * @param[in] n output data size. Both index arrays should be the same size as
 * this
 * @param[inout] out_data ouput data array
 */
ABYSS_EXPORT void add(const int32_t* in_data1, const size_t* id1,
                      const int32_t* in_data2, const size_t* id2,
                      const size_t n, int32_t* out_data);
ABYSS_EXPORT void add(const double* in_data1, const size_t* id1,
                      const int32_t* in_data2, const size_t* id2,
                      const size_t n, double* out_data);
ABYSS_EXPORT void add(const int32_t* in_data1, const size_t* id1,
                      const double* in_data2, const size_t* id2, const size_t n,
                      double* out_data);
ABYSS_EXPORT void add(const double* in_data1, const size_t* id1,
                      const double* in_data2, const size_t* id2, const size_t n,
                      double* out_data);

// ABYSS_EXPORT void sub(const int32_t* in1, const int32_t* in2, const size_t& n,
//                       int32_t* out) noexcept;
// ABYSS_EXPORT void sub(const double* in1, const int32_t* in2, const size_t& n,
//                       double* out) noexcept;
// ABYSS_EXPORT void sub(const int32_t* in1, const double* in2, const size_t& n,
//                       double* out) noexcept;
// ABYSS_EXPORT void sub(const double* in1, const double* in2, const size_t& n,
//                       double* out) noexcept;

ABYSS_EXPORT void sub(const int32_t* in_data1, const size_t* id1,
                      const int32_t* in_data2, const size_t* id2,
                      const size_t n, int32_t* out_data);
ABYSS_EXPORT void sub(const double* in_data1, const size_t* id1,
                      const int32_t* in_data2, const size_t* id2,
                      const size_t n, double* out_data);
ABYSS_EXPORT void sub(const int32_t* in_data1, const size_t* id1,
                      const double* in_data2, const size_t* id2, const size_t n,
                      double* out_data);
ABYSS_EXPORT void sub(const double* in_data1, const size_t* id1,
                      const double* in_data2, const size_t* id2, const size_t n,
                      double* out_data);

// ABYSS_EXPORT void mult(const int32_t* in1, const int32_t* in2, const size_t& n,
//                        int32_t* out) noexcept;
// ABYSS_EXPORT void mult(const int32_t* in1, const double* in2, const size_t& n,
//                        double* out) noexcept;
// ABYSS_EXPORT void mult(const double* in1, const int* in2, const size_t& n,
//                        double* out) noexcept;
// ABYSS_EXPORT void mult(const double* in1, const double* in2, const size_t& n,
//                        double* out) noexcept;

ABYSS_EXPORT void mult(const int32_t* in_data1, const size_t* id1,
                       const int32_t* in_data2, const size_t* id2,
                       const size_t n, int32_t* out_data);
ABYSS_EXPORT void mult(const double* in_data1, const size_t* id1,
                       const int32_t* in_data2, const size_t* id2,
                       const size_t n, double* out_data);
ABYSS_EXPORT void mult(const int32_t* in_data1, const size_t* id1,
                       const double* in_data2, const size_t* id2,
                       const size_t n, double* out_data);
ABYSS_EXPORT void mult(const double* in_data1, const size_t* id1,
                       const double* in_data2, const size_t* id2,
                       const size_t n, double* out_data);

// ABYSS_EXPORT void div(const int32_t* in1, const int32_t* in2, const size_t& n,
//                       int32_t* out) noexcept;
// ABYSS_EXPORT void div(const int32_t* in1, const double* in2, const size_t& n,
//                       double* out) noexcept;
// ABYSS_EXPORT void div(const double* in1, const int32_t* in2, const size_t& n,
//                       double* out) noexcept;
// ABYSS_EXPORT void div(const double* in1, const double* in2, const size_t& n,
//                       double* out) noexcept;

ABYSS_EXPORT void div(const int32_t* in_data1, const size_t* id1,
                      const int32_t* in_data2, const size_t* id2,
                      const size_t n, int32_t* out_data);
ABYSS_EXPORT void div(const double* in_data1, const size_t* id1,
                      const int32_t* in_data2, const size_t* id2,
                      const size_t n, double* out_data);
ABYSS_EXPORT void div(const int32_t* in_data1, const size_t* id1,
                      const double* in_data2, const size_t* id2, const size_t n,
                      double* out_data);
ABYSS_EXPORT void div(const double* in_data1, const size_t* id1,
                      const double* in_data2, const size_t* id2, const size_t n,
                      double* out_data);

/**
 * @brief experimental interface
 *
 * these functions uses axpy blas routines, don't know if they are faster.
 */
template <typename T1, typename T2>
void xadd(const T1* in1, const T2* in2, const size_t& n,
          std::common_type_t<T1, T2>* out) {
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


ABYSS_EXPORT void neg(const uint8_t* in_data, const size_t* ids, const size_t n,
                      uint8_t* out_data);
ABYSS_EXPORT void neg(const int32_t* in_data, const size_t* ids, const size_t n,
                      int32_t* out_data);
ABYSS_EXPORT void neg(const double* in_data, const size_t* ids, const size_t n,
                      double* out_data);

}  // namespace abyss::backend

#endif