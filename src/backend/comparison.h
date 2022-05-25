#ifndef ABYSS_BACKEND_COMPARISON_H
#define ABYSS_BACKEND_COMPARISON_H

#include <cstddef>
#include <cstdint>

#include "abyss_export.h"

namespace abyss::backend {

ABYSS_EXPORT void equal(const int32_t* in1, const int32_t* in2, const size_t& n,
                        bool* out) noexcept;
ABYSS_EXPORT void equal(const int32_t* in1, const double* in2, const size_t& n,
                        bool* out) noexcept;
ABYSS_EXPORT void equal(const double* in1, const int32_t* in2, const size_t& n,
                        bool* out) noexcept;
ABYSS_EXPORT void equal(const double* in1, const double* in2, const size_t& n,
                        bool* out) noexcept;

ABYSS_EXPORT void equal(const int32_t* in_data1, const size_t* id1,
                        const int32_t* in_data2, const size_t* id2,
                        const size_t n, bool* out_data);
ABYSS_EXPORT void equal(const double* in_data1, const size_t* id1,
                        const int32_t* in_data2, const size_t* id2,
                        const size_t n, bool* out_data);
ABYSS_EXPORT void equal(const int32_t* in_data1, const size_t* id1,
                        const double* in_data2, const size_t* id2,
                        const size_t n, bool* out_data);
ABYSS_EXPORT void equal(const double* in_data1, const size_t* id1,
                        const double* in_data2, const size_t* id2,
                        const size_t n, bool* out_data);

ABYSS_EXPORT void not_equal(const int32_t* in1, const int32_t* in2,
                            const size_t& n, bool* out) noexcept;
ABYSS_EXPORT void not_equal(const int32_t* in1, const double* in2,
                            const size_t& n, bool* out) noexcept;
ABYSS_EXPORT void not_equal(const double* in1, const int32_t* in2,
                            const size_t& n, bool* out) noexcept;
ABYSS_EXPORT void not_equal(const double* in1, const double* in2,
                            const size_t& n, bool* out) noexcept;

ABYSS_EXPORT void not_equal(const int32_t* in_data1, const size_t* id1,
                            const int32_t* in_data2, const size_t* id2,
                            const size_t n, bool* out_data);
ABYSS_EXPORT void not_equal(const double* in_data1, const size_t* id1,
                            const int32_t* in_data2, const size_t* id2,
                            const size_t n, bool* out_data);
ABYSS_EXPORT void not_equal(const int32_t* in_data1, const size_t* id1,
                            const double* in_data2, const size_t* id2,
                            const size_t n, bool* out_data);
ABYSS_EXPORT void not_equal(const double* in_data1, const size_t* id1,
                            const double* in_data2, const size_t* id2,
                            const size_t n, bool* out_data);
// ABYSS_EXPORT void greater_than(const int32_t* in1, const int32_t* in2, const
// size_t& n,
//                bool* out) noexcept;
// ABYSS_EXPORT void greater_than(const int32_t* in1, const double* in2, const
// size_t& n,
//                bool* out) noexcept;
// ABYSS_EXPORT void greater_than(const double* in1, const int32_t* in2, const
// size_t& n,
//                bool* out) noexcept;
// ABYSS_EXPORT void greater_than(const double* in1, const double* in2, const
// size_t& n,
//                bool* out) noexcept;

// ABYSS_EXPORT void less_than(const int32_t* in1, const int32_t* in2, const
// size_t& n,
//                bool* out) noexcept;
// ABYSS_EXPORT void less_than(const int32_t* in1, const double* in2, const
// size_t& n,
//                bool* out) noexcept;
// ABYSS_EXPORT void less_than(const double* in1, const int32_t* in2, const
// size_t& n,
//                bool* out) noexcept;
// ABYSS_EXPORT void less_than(const double* in1, const double* in2, const
// size_t& n,
//                bool* out) noexcept;
}  // namespace abyss::backend

#endif