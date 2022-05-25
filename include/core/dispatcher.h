#ifndef ABYSS_CORE_DISPATCHER_H
#define ABYSS_CORE_DISPATCHER_H

/**
 * @file dispatcher.h
 * 
 * Dispatchers are used as adaptors to call the accept methods on their data.
 * This extra layer of redirection is necessary so users do not use our visitors directly.
 */


#include <iostream>

#include "core/visitor.h"
#include "core/traits.h"

namespace abyss::core {

/**
 * @brief a decorator to expose the visitor interface
 */
// template <typename T>
// class Dispatcher : public T, public Visitable {
//  public:
//   static_assert(std::is_base_of<Dispatchable, T>::value, "T should be a dispatchable type");

//   Dispatcher(T parent) : T(parent) {}
//   virtual ~Dispatcher() = default;

//   using T::desc;
  
//   void accept(VisitorBase* vis) override {
//     T::data()->accept(vis);
//   }

//   void accept(VisitorBase* vis, Visitable* b) override {
//     // std::cout<<"dispatch accept (meta) > ";
//     // b->accept(vis, T::data());
//     T::data()->accept(vis, b);
//   }

//   void accept(VisitorBase* vis, ArrayImpl<bool>* a) override {
//     T::data()->accept(vis, a);
//   }
//   void accept(VisitorBase* vis, ArrayImpl<uint8_t>* a) override {
//     T::data()->accept(vis, a);
//   }
//   void accept(VisitorBase* vis, ArrayImpl<int32_t>* a) override {
//     // std::cout<<"dispatch accept > ";
//     T::data()->accept(vis, a);
//   }
//   void accept(VisitorBase* vis, ArrayImpl<double>* a) override {
//     T::data()->accept(vis, a);
//   }
// };

/**
 * @brief dispatcher for data centric classes
 */
template <typename T>
class DataDispatcher : public T, public Visitable {
 public:
  // static_assert(std::is_base_of<Tensor, T>::value, "T should be a dispatchable type");

  DataDispatcher(T parent) : T(parent) {}
  virtual ~DataDispatcher() = default;

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
};


/**
 * @brief dispatcher for type centric classes
 */
template <typename T>
class TypeDispatcher : public T, public Visitable {
 public:
  // static_assert(std::is_base_of<ScalarType, T>::value, "T should be a dispatchable type");

  TypeDispatcher(T parent) : T(parent) {}
  virtual ~TypeDispatcher() = default;

  void accept(VisitorBase* vis) override {
    T::type()->accept(vis);
  }

  void accept(VisitorBase* vis, Visitable* b) override {
    // std::cout<<"dispatch accept (meta) > ";
    // b->accept(vis, T::data());
    T::type()->accept(vis, b);
  }

  void accept(VisitorBase* vis, ArrayImpl<bool>* a) override {
    T::type()->accept(vis, a);
  }
  void accept(VisitorBase* vis, ArrayImpl<uint8_t>* a) override {
    T::type()->accept(vis, a);
  }
  void accept(VisitorBase* vis, ArrayImpl<int32_t>* a) override {
    // std::cout<<"dispatch accept > ";
    T::type()->accept(vis, a);
  }
  void accept(VisitorBase* vis, ArrayImpl<double>* a) override {
    T::type()->accept(vis, a);
  }
};

}  // namespace abyss::core

#endif