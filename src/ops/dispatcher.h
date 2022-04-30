#ifndef ABYSS_CORE_DISPATCHER_H
#define ABYSS_CORE_DISPATCHER_H

#include <iostream>

#include "core/visitor.h"
#include "core/traits.h"

namespace abyss::core {

/**
 * @brief a decorator to expose the visitor interface
 */
template <typename T>
class Dispatcher : public T, public Visitable {
 public:
  static_assert(std::is_base_of<Dispatchable, T>::value, "T should be a dispatchable type");

  Dispatcher(T parent) : T(parent) {}
  virtual ~Dispatcher() = default;

  using T::desc;
  
  void accept(VisitorBase* vis) override {
    T::data()->accept(vis);
  }

  void accept(VisitorBase* vis, Visitable* b) override {
    // std::cout<<"dispatch accept (meta) > ";
    // b->accept(vis, T::data());
    T::data()->accept(vis, b);
  }

  void accept(VisitorBase* vis, ArrayImpl<bool>* a) override {
    T::data()->accept(vis, a);
  }
  void accept(VisitorBase* vis, ArrayImpl<uint8_t>* a) override {
    T::data()->accept(vis, a);
  }
  void accept(VisitorBase* vis, ArrayImpl<int32_t>* a) override {
    // std::cout<<"dispatch accept > ";
    T::data()->accept(vis, a);
  }
  void accept(VisitorBase* vis, ArrayImpl<double>* a) override {
    T::data()->accept(vis, a);
  }

 private:
  // T parent_;
//   void pre_accept();
};
}  // namespace abyss::core

#endif