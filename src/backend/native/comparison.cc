#include "comparison.h"

namespace abyss::backend {

void equal(const int32_t* in1, const int32_t* in2, const size_t& n, bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] == in2[i];
  }
}
void equal(const int32_t* in1, const double* in2, const size_t& n, bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] == in2[i];
  }
}
void equal(const double* in1, const int32_t* in2, const size_t& n, bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] == in2[i];
  }
}
void equal(const double* in1, const double* in2, const size_t& n, bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] == in2[i];
  }
}

void not_equal(const int32_t* in1, const int32_t* in2, const size_t& n, bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] != in2[i];
  }
}
void not_equal(const int32_t* in1, const double* in2, const size_t& n, bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] != in2[i];
  }
}
void not_equal(const double* in1, const int32_t* in2, const size_t& n, bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] != in2[i];
  }
}
void not_equal(const double* in1, const double* in2, const size_t& n, bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] != in2[i];
  }
}

}