#ifndef ABYSS_CORE_DTYPE_H
#define ABYSS_CORE_DTYPE_H

#include <complex>
#include <exception>
#include <typeindex>

#include "array.h"
#include "visitor.h"
// #include "expression.h"
#include "traits.h"

namespace abyss::core {

struct DTypeBase : public Visitable {
  virtual ~DTypeBase() = default;

  virtual std::type_index id() const = 0;
  virtual size_t itemsize() const = 0;
};

template <typename T>
class DTypeImpl : public DTypeBase {
 public:
  using value_type = T;

  DTypeImpl() = default;

  std::type_index id() const override { return typeid(T); }
  size_t itemsize() const override { return 0; }

//  protected:
  void accept(VisitorBase* vis) override {
    auto visitor = dynamic_cast<UnaryVisitor<DTypeImpl<T>>*>(vis);
    visitor->visit(this);
  }

  void accept(VisitorBase* vis, Visitable* b) override { b->accept(vis, this); }
  void accept(VisitorBase* vis, ArrayImpl<bool>* a) override {
    auto visitor =
        dynamic_cast<BinaryVisitor<ArrayImpl<bool>, DTypeImpl<T>>*>(vis);
    visitor->visit(a, this);
  }
  void accept(VisitorBase* vis, ArrayImpl<uint8_t>* a) override {
    auto visitor =
        dynamic_cast<BinaryVisitor<ArrayImpl<uint8_t>, DTypeImpl<T>>*>(vis);
    visitor->visit(a, this);
  }
  void accept(VisitorBase* vis, ArrayImpl<int32_t>* a) override {
    auto visitor =
        dynamic_cast<BinaryVisitor<ArrayImpl<int32_t>, DTypeImpl<T>>*>(vis);
    visitor->visit(a, this);
  }
  void accept(VisitorBase* vis, ArrayImpl<double>* a) override {
    auto visitor =
        dynamic_cast<BinaryVisitor<ArrayImpl<double>, DTypeImpl<T>>*>(vis);
    visitor->visit(a, this);
  }
};

template <>
class DTypeImpl<void> : public DTypeBase {
 public:
  using value_type = void;

  DTypeImpl() = default;

  std::type_index id() const override { return typeid(void); }
  size_t itemsize() const override { return 0; }

//  protected:
  void accept(VisitorBase* vis) override {
    auto visitor = dynamic_cast<UnaryVisitor<DTypeImpl<void>>*>(vis);
    visitor->visit(this);
  }

  void accept(VisitorBase* vis, Visitable* b) override { b->accept(vis, this); }
  void accept(VisitorBase* vis, ArrayImpl<bool>* a) override {
    auto visitor =
        dynamic_cast<BinaryVisitor<ArrayImpl<bool>, DTypeImpl<void>>*>(vis);
    visitor->visit(a, this);
  }
  void accept(VisitorBase* vis, ArrayImpl<uint8_t>* a) override {
    auto visitor =
        dynamic_cast<BinaryVisitor<ArrayImpl<uint8_t>, DTypeImpl<void>>*>(vis);
    visitor->visit(a, this);
  }
  void accept(VisitorBase* vis, ArrayImpl<int32_t>* a) override {
    auto visitor =
        dynamic_cast<BinaryVisitor<ArrayImpl<int32_t>, DTypeImpl<void>>*>(vis);
    visitor->visit(a, this);
  }
  void accept(VisitorBase* vis, ArrayImpl<double>* a) override {
    auto visitor =
        dynamic_cast<BinaryVisitor<ArrayImpl<double>, DTypeImpl<void>>*>(vis);
    visitor->visit(a, this);
  }
};

// static DType* kNone = kNone_;
// static DType* kUint8 = kUint8_;
// static DType* kInt32 = kInt32_;
// static DType* kDouble = kDouble_;
// static DType* kComplex128 = kComplex128_;

// bool operator==(DType* a, DType* b) {
//   return a.id() == b.id();
// }

// bool operator!=(DType* a, DType* b) {
//   return !(a == b);
// }

}  // namespace abyss::core

#endif