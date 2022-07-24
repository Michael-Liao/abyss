#include "reduction.h"

namespace abyss::backend {
void all() {}

void sum(const bool* in_data, int stride, size_t n, bool* out_data) {
  for (size_t i = 0; i < n; i++) {
    *out_data += in_data[i * stride];
  }
}
void sum(const uint8_t* in_data, int stride, size_t n, uint8_t* out_data) {
  for (size_t i = 0; i < n; i++) {
    *out_data += in_data[i * stride];
  }
}
void sum(const int32_t* in_data, int stride, size_t n, int32_t* out_data) {
  for (size_t i = 0; i < n; i++) {
    *out_data += in_data[i * stride];
  }
}
void sum(const double* in_data, int stride, size_t n, double* out_data) {
  for (size_t i = 0; i < n; i++) {
    *out_data += in_data[i * stride];
  }
}
}  // namespace abyss::backend