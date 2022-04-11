#include "arithmetics.h"

#include <cblas.h>

#include <algorithm>
#include <cstdlib>
// #include "types.h"
// #include "core/array.h"

namespace abyss::backend {

void add(const int32_t* in1, const int32_t* in2, const size_t& n,
         int32_t* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] + in2[i];
  }
}
void add(const double* in1, const int32_t* in2, const size_t& n,
         double* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] + in2[i];
  }
}
void add(const int32_t* in1, const double* in2, const size_t& n,
         double* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] + in2[i];
  }
}
void add(const double* in1, const double* in2, const size_t& n,
         double* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] + in2[i];
  }
}

void sub(const int32_t* in1, const int32_t* in2, const size_t& n,
         int32_t* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] - in2[i];
  }
}
void sub(const double* in1, const int32_t* in2, const size_t& n,
         double* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] - in2[i];
  }
}
void sub(const int32_t* in1, const double* in2, const size_t& n,
         double* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] - in2[i];
  }
}
void sub(const double* in1, const double* in2, const size_t& n,
         double* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] * in2[i];
  }
}

void mult(const int32_t* in1, const int32_t* in2, const size_t& n,
          int32_t* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] * in2[i];
  }
}
void mult(const int32_t* in1, const double* in2, const size_t& n,
          double* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] * in2[i];
  }
}
void mult(const double* in1, const int32_t* in2, const size_t& n,
          double* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] * in2[i];
  }
}
void mult(const double* in1, const double* in2, const size_t& n,
          double* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] * in2[i];
  }
}

void div(const int32_t* in1, const int32_t* in2, const size_t& n,
         int32_t* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] / in2[i];
  }
}
void div(const int32_t* in1, const double* in2, const size_t& n,
         double* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] / in2[i];
  }
}
void div(const double* in1, const int32_t* in2, const size_t& n,
         double* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] / in2[i];
  }
}
void div(const double* in1, const double* in2, const size_t& n,
         double* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] / in2[i];
  }
}

// void xsub(const int* in1, const int* in2, const size_t& n, int* out) noexcept
// {
//   using target_t = float;
//   target_t* buffer = (target_t*)std::malloc(2 * n * sizeof(target_t));

//   target_t* out_f = buffer;
//   target_t* in2_f = buffer + n;
//   std::copy_n(in1, n, out_f);
//   std::copy_n(in2, n, in2_f);

//   cblas_saxpy(n, -1, in2_f, 1, out_f, 1);

//   std::copy_n(out_f, n, out);

//   std::free(buffer);
// }
// void xsub(const double* in1, const int* in2, const size_t& n,
//          double* out) noexcept {
//   using target_t = double;
//   target_t* buffer = (target_t*)std::malloc(n * sizeof(target_t));

//   target_t* in2_f = buffer;
//   std::copy_n(in1, n, out);
//   std::copy_n(in2, n, in2_f);

//   cblas_daxpy(n, -1, in2_f, 1, out, 1);

//   std::free(buffer);
// }
// void xsub(const int* in1, const double* in2, const size_t& n,
//          double* out) noexcept {
//   std::copy_n(in1, n, out);
//   cblas_daxpy(n, -1, in2, 1, out, 1);
// }
// void xsub(const double* in1, const double* in2, const size_t& n,
//          double* out) noexcept {
//   std::copy_n(in1, n, out);
//   cblas_daxpy(n, -1, in2, 1, out, 1);
// }

}  // namespace abyss::backend