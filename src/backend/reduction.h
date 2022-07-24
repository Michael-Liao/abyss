#ifndef ABYSS_BACKEND_REDUCTION_H
#define ABYSS_BACKEND_REDUCTION_H

#include <cstddef>
#include <cstdint>

#include "abyss_export.h"

namespace abyss::backend {
ABYSS_EXPORT void all();

ABYSS_EXPORT void sum(const bool* in_data, int stride, size_t n,
                      bool* out_data);
ABYSS_EXPORT void sum(const uint8_t* in_data, int stride, size_t n,
                      uint8_t* out_data);
ABYSS_EXPORT void sum(const int32_t* in_data, int stride, size_t n,
                      int32_t* out_data);
ABYSS_EXPORT void sum(const double* in_data, int stride, size_t n,
                      double* out_data);
}  // namespace abyss::backend

#endif