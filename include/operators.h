#ifndef ABYSS_OPERATORS_H
#define ABYSS_OPERATORS_H

#include "abyss_export.h"
#include "tensor.h"


namespace abyss {

ABYSS_EXPORT Tensor operator+(Tensor a, Tensor b);
ABYSS_EXPORT Tensor operator-(Tensor a, Tensor b);
ABYSS_EXPORT Tensor operator*(Tensor a, Tensor b);
ABYSS_EXPORT Tensor operator/(Tensor a, Tensor b);

ABYSS_EXPORT Tensor operator==(Tensor a, Tensor b);
ABYSS_EXPORT Tensor operator==(Tensor a, bool b) { return a == Tensor(b); }
ABYSS_EXPORT Tensor operator==(bool a, Tensor b) { return Tensor(a) == b; }
ABYSS_EXPORT Tensor operator==(Tensor a, uint8_t b) { return a == Tensor(b); }
ABYSS_EXPORT Tensor operator==(uint8_t a, Tensor b) { return Tensor(a) == b; }
ABYSS_EXPORT Tensor operator==(Tensor a, int32_t b) { return a == Tensor(b); }
ABYSS_EXPORT Tensor operator==(int32_t a, Tensor b) { return Tensor(a) == b; }
ABYSS_EXPORT Tensor operator==(Tensor a, double b) { return a == Tensor(b); }
ABYSS_EXPORT Tensor operator==(double a, Tensor b) { return Tensor(a) == b; }

ABYSS_EXPORT Tensor operator!=(Tensor a, Tensor b);
ABYSS_EXPORT Tensor operator!=(Tensor a, bool b) { return a != Tensor(b); }
ABYSS_EXPORT Tensor operator!=(bool a, Tensor b) { return Tensor(a) != b; }
ABYSS_EXPORT Tensor operator!=(Tensor a, uint8_t b) { return a != Tensor(b); }
ABYSS_EXPORT Tensor operator!=(uint8_t a, Tensor b) { return Tensor(a) != b; }
ABYSS_EXPORT Tensor operator!=(Tensor a, int32_t b) { return a != Tensor(b); }
ABYSS_EXPORT Tensor operator!=(int32_t a, Tensor b) { return Tensor(a) != b; }
ABYSS_EXPORT Tensor operator!=(Tensor a, double b) { return a != Tensor(b); }
ABYSS_EXPORT Tensor operator!=(double a, Tensor b) { return Tensor(a) != b; }

}  // namespace abyss

#endif