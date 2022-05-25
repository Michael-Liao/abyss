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
                        public UnaryVisitor<ArrayImpl<TgtTp>> {
 public:
  ToScalarVisitor(ArrayDesc desc) : desc_{desc} {}

  void visit(ArrayImpl<TgtTp>* a) override { eval(a); }

  TgtTp value() const { return value_; }

 private:
  TgtTp value_;
  ArrayDesc desc_;

  // template <typename InTp>
  void eval(ArrayImpl<TgtTp>* a) {
    if (desc_.shape.size() != 1 || desc_.shape[0] != 1) {
      throw std::domain_error(
          "array of element more than one cannot be converted to scalar");
    }
    // if (a->size() != 1) {
    //   throw std::runtime_error(
    //       "array of element more than one cannot be converted to scalar");
    // }

    value_ = a->at(desc_.offset);
    // value_ = 0;
  }
};

// class AstypeVisitor;

}  // namespace abyss::core

#endif