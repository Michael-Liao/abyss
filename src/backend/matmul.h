#ifndef ABYSS_BACKEND_NATIVE_MATMUL_H
#define ABYSS_BACKEND_NATIVE_MATMUL_H

// #include <cblas-openblas.h>
#include <cstddef>
#include <cstdint>

#include "abyss_export.h"
// #include "backend/backend.h"
// #include "tensor.h"

namespace abyss::backend {

void matmul(const int* A, const float* B, int m, int k, int n,
            float* C) noexcept;


ABYSS_EXPORT void matmul(const int32_t* A, const int32_t* B, int m, int k, int n,
            int32_t* C) noexcept;

ABYSS_EXPORT void matmul(const int32_t* A, const double* B, int m, int k, int n,
            double* C) noexcept;
ABYSS_EXPORT void matmul(const double* A, const int32_t* B, int m, int k, int n,
            double* C) noexcept;
ABYSS_EXPORT void matmul(const double* A, const double* B, int m, int k, int n,
            double* C) noexcept;

}  // namespace abyss::backend
#endif