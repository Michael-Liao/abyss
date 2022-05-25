#include "comparison.h"

namespace abyss::backend {

void equal(const int32_t* in1, const int32_t* in2, const size_t& n,
           bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] == in2[i];
  }
}
void equal(const int32_t* in1, const double* in2, const size_t& n,
           bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] == in2[i];
  }
}
void equal(const double* in1, const int32_t* in2, const size_t& n,
           bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] == in2[i];
  }
}
void equal(const double* in1, const double* in2, const size_t& n,
           bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] == in2[i];
  }
}

void equal(const int32_t* in_data1, const size_t* id1, const int32_t* in_data2,
           const size_t* id2, const size_t n, bool* out_data) {
  for (size_t i = 0; i < n; i++) {
    out_data[i] = in_data1[id1[i]] == in_data2[id2[i]];
  }
}
void equal(const double* in_data1, const size_t* id1, const int32_t* in_data2,
           const size_t* id2, const size_t n, bool* out_data) {
  for (size_t i = 0; i < n; i++) {
    out_data[i] = in_data1[id1[i]] == in_data2[id2[i]];
  }
}
void equal(const int32_t* in_data1, const size_t* id1, const double* in_data2,
           const size_t* id2, const size_t n, bool* out_data) {
  for (size_t i = 0; i < n; i++) {
    out_data[i] = in_data1[id1[i]] == in_data2[id2[i]];
  }
}
void equal(const double* in_data1, const size_t* id1, const double* in_data2,
           const size_t* id2, const size_t n, bool* out_data) {
  for (size_t i = 0; i < n; i++) {
    out_data[i] = in_data1[id1[i]] == in_data2[id2[i]];
  }
}

void not_equal(const int32_t* in1, const int32_t* in2, const size_t& n,
               bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] != in2[i];
  }
}
void not_equal(const int32_t* in1, const double* in2, const size_t& n,
               bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] != in2[i];
  }
}
void not_equal(const double* in1, const int32_t* in2, const size_t& n,
               bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] != in2[i];
  }
}
void not_equal(const double* in1, const double* in2, const size_t& n,
               bool* out) noexcept {
  for (size_t i = 0; i < n; i++) {
    out[i] = in1[i] != in2[i];
  }
}

void not_equal(const int32_t* in_data1, const size_t* id1,
               const int32_t* in_data2, const size_t* id2, const size_t n,
               bool* out_data) {
  for (size_t i = 0; i < n; i++) {
    out_data[i] = in_data1[id1[i]] != in_data2[id2[i]];
  }
}
void not_equal(const double* in_data1, const size_t* id1,
               const int32_t* in_data2, const size_t* id2, const size_t n,
               bool* out_data) {
  for (size_t i = 0; i < n; i++) {
    out_data[i] = in_data1[id1[i]] != in_data2[id2[i]];
  }
}
void not_equal(const int32_t* in_data1, const size_t* id1,
               const double* in_data2, const size_t* id2, const size_t n,
               bool* out_data) {
  for (size_t i = 0; i < n; i++) {
    out_data[i] = in_data1[id1[i]] != in_data2[id2[i]];
  }
}
void not_equal(const double* in_data1, const size_t* id1,
               const double* in_data2, const size_t* id2, const size_t n,
               bool* out_data) {
  for (size_t i = 0; i < n; i++) {
    out_data[i] = in_data1[id1[i]] != in_data2[id2[i]];
  }
}

}  // namespace abyss::backend