#ifndef ABYSS_CORE_VISITOR_H
#define ABYSS_CORE_VISITOR_H

/**
 * @file
 * Common definitions for backend functions and utilities.
 */

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "types.h"


namespace abyss::core {

class Array;

template <typename T>
class ArrayImpl;

/**
 * @brief Common base class for all visitors
 *
 * It stores members required to build a Tensor.
 */
// class VisitorBase : public Tensor {
class VisitorBase {
 public:
  virtual ~VisitorBase() {}

//   Tensor result() const {
//     Tensor out;
//     out.dtype_ = result_dtype_;
//     out.shape_ = output_shape_;
//     out.strides_ = output_strides_;
//     out.data_ = shared_data_;

//     return out;
//   }

//  protected:
//   ScalarType result_dtype_;
//   std::vector<int> output_shape_;
//   std::vector<int> output_strides_;
//   std::shared_ptr<Array> shared_data_;
};

/**
 * @brief Visitable type property
 *
 * Any type that supports visit should inherit from this type and implement all
 * `accept` functions.
 */
class Visitable {
 public:
  // accepts unary functions
  virtual void accept(VisitorBase*) = 0;

  // accepts binary functions (triple dispatch)
  virtual void accept(VisitorBase*, Visitable*) = 0;
  virtual void accept(VisitorBase*, ArrayImpl<bool>*) = 0;
  virtual void accept(VisitorBase*, ArrayImpl<uint8_t>*) = 0;
  virtual void accept(VisitorBase*, ArrayImpl<int32_t>*) = 0;
  virtual void accept(VisitorBase*, ArrayImpl<double>*) = 0;
};

/**
 * @brief Visitor for unary operations.
 *
 * This is used for decoupling the `visit` method for each type.
 */
template <typename T>
class UnaryVisitor {
 public:
  virtual void visit(T* input) = 0;
};

/**
 * @brief Visitor for binary operations.
 *
 * Most operations requires visitation of 2 arrays.
 */
template <typename T1, typename T2>
class BinaryVisitor {
 public:
  virtual void visit(T1* a, T2* b) = 0;
};

// template <typename T1, typename T2>
// class InplaceBinaryVisitor {
//  public:
//   virtual void visit(T1* in, T2* in_out) = 0;
// };

/**
 * The ultimate visitor that take multiple arguments.
 * There is no accept stategy to work with this. abandoned
 */
// template <typename... VisitableTps>
// class XVisitor {
//  public:
//   virtual void visit(VisitableTps*... visitable_types) = 0;
// };

}  // namespace abyss::core

#endif