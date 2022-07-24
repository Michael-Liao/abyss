#include "amath.h"

namespace abyss::backend {
void exp(const uint8_t* in_data, const size_t* ids, const size_t n,
         uint8_t* out_data) {
  for (size_t i = 0; i < n; i++) {
    out_data[ids[i]] = std::exp(in_data[ids[i]]);
  }
}
void exp(const int32_t* in_data, const size_t* ids, const size_t n,
         int32_t* out_data) {
  for (size_t i = 0; i < n; i++) {
    out_data[ids[i]] = std::exp(in_data[ids[i]]);
  }
}
void exp(const double* in_data, const size_t* ids, const size_t n,
         double* out_data) {
  for (size_t i = 0; i < n; i++) {
    out_data[ids[i]] = std::exp(in_data[ids[i]]);
  }
}

void log(const uint8_t* in_data, const size_t* ids, const size_t n,
         uint8_t* out_data) {
  for (size_t i = 0; i < n; i++) {
    out_data[ids[i]] = std::log(in_data[ids[i]]);
  }
}
void log(const int32_t* in_data, const size_t* ids, const size_t n,
         int32_t* out_data) {
  for (size_t i = 0; i < n; i++) {
    out_data[ids[i]] = std::log(in_data[ids[i]]);
  }
}
void log(const double* in_data, const size_t* ids, const size_t n,
         double* out_data) {
  for (size_t i = 0; i < n; i++) {
    out_data[ids[i]] = std::log(in_data[ids[i]]);
  }
}
}  // namespace abyss::backend