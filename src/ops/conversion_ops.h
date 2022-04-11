#ifndef ABYSS_CORE_CONVERSION_OPS_H
#define ABYSS_CORE_CONVERSION_OPS_H

#include <cstdint>
#include <exception>
#include <vector>

#include "core/array.h"
#include "core/visitor.h"

namespace abyss::core {

template <typename TgtTp>
class ToScalarVisitor : public VisitorBase,
                        public UnaryVisitor<ArrayImpl<bool>>,
                        public UnaryVisitor<ArrayImpl<uint8_t>>,
                        public UnaryVisitor<ArrayImpl<int32_t>>,
                        public UnaryVisitor<ArrayImpl<double>> {
 public:
  ToScalarVisitor() = default;

  void visit(ArrayImpl<bool>* a) override { eval(a); }
  void visit(ArrayImpl<uint8_t>* a) override { eval(a); }
  void visit(ArrayImpl<int32_t>* a) override { eval(a); }
  void visit(ArrayImpl<double>* a) override { eval(a); }

  TgtTp value() const { return value_; }

 private:
  TgtTp value_;

  template <typename InTp>
  void eval(ArrayImpl<InTp>* a) {
    if (a->size() != 1) {
      throw std::runtime_error(
          "array of element more than one cannot be converted to scalar");
    }

    value_ = static_cast<TgtTp>(a->at(0));
    // value_ = 0;
  }
};

// class AstypeVisitor;

}  // namespace abyss::core

#endif