#include "matmul.h"

#include <cblas.h>
#include <algorithm>

namespace abyss::backend {

/**
 * @brief matrix multiplication wrapper for BLAS
 *
 * This function assumes continuous storage.
 * Since BLAS routines only works with floating point values,
 * none-floating arrays have to be copied in advance.
 */
void matmul(int* A, float* B, int m, int k, int n, float* C) noexcept {
  float* A_copy = (float*)std::malloc(m * k);
  for (int i = 0; i < m * k; i++) {
    A_copy[i] = A[i];  // convert to float
  }

  // if (m == n && k == 1) {  // vector to vector
  //   cblas_sdot(m, A_copy, 1, B, 1);
  // } else if (n == 1) {  // matrix to vector
  //   cblas_sgemv(CblasColMajor, CblasTrans, m, k, 1.0f, A_copy, k, B, 1, 0.0f,
  //   C,
  //               1);
  // } else {  // matrix to matrix
  //   cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, B,
  //   k,
  //               A_copy, m, 0.0f, C, m);
  // }

  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, B, k,
              A_copy, m, 0.0f, C, m);
  std::free(A_copy);
}

void matmul(const int32_t* A, const int32_t* B, int m, int k, int n,
            int32_t* C) noexcept {
  using target_t = float;

  const size_t A_size = m * k;
  const size_t B_size = k * n;
  const size_t C_size = m * n;
  
  target_t* buffer = (target_t*)std::malloc((A_size + B_size + C_size) * sizeof(target_t));
  if (!buffer) return; // malloc returned nullptr

  target_t* A_f = buffer;
  target_t* B_f = buffer + A_size;
  target_t* C_f = buffer + A_size + B_size;

  std::copy_n(A, A_size, A_f);
  std::copy_n(B, B_size, B_f);

  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, k, 1.0f, B_f, n,
              A_f, k, 0.0f, C_f, n);

  std::copy_n(C_f, C_size, C);

  std::free(buffer);
}

void matmul(const int32_t* A, const double* B, int m, int k, int n,
            double* C) noexcept {
  using target_t = double;

  size_t A_size = m * k;
  size_t B_size = k * n;
  size_t C_size = m * n;
  
  target_t* buffer = (target_t*)std::malloc(A_size * sizeof(target_t));
  if (!buffer) return; // malloc returned nullptr

  target_t* A_f = buffer;

  std::copy_n(A, A_size, A_f);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, k, 1.0f, B, n,
              A_f, k, 0.0f, C, n);

  std::free(buffer);
}

void matmul(const double* A, const int32_t* B, int m, int k, int n,
            double* C) noexcept {
  using target_t = double;

  size_t A_size = m * k;
  size_t B_size = k * n;
  size_t C_size = m * n;
  
  target_t* buffer = (target_t*)std::malloc(A_size * sizeof(target_t));
  if (!buffer) return; // malloc returned nullptr

  target_t* B_f = buffer;

  std::copy_n(B, B_size, B_f);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, k, 1.0f, B_f, n,
              A, k, 0.0f, C, n);

  std::free(buffer);
}

void matmul(const double* A, const double* B, int m, int k, int n,
            double* C) noexcept {
  // don't need any conversions, just a simple wrapper
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, k, 1.0f, B, n,
              A, k, 0.0f, C, n);
}

}  // namespace abyss::backend