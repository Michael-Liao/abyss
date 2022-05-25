#ifndef ABYSS_AUTOGRAD_FUNCTIONS_H
#define ABYSS_AUTOGRAD_FUNCTIONS_H
/**
 * This file is for defining functions that could do autograd
 */
#include <array>
#include <ostream>
#include <string>
#include <vector>

#include "abyss_export.h"
#include "functional.h"
#include "graph.h"
#include "operators.h"
#include "tensor.h"

namespace abyss::autograd {
/**
 * @brief The function interface
 */
template <typename ChildType>
class ABYSS_EXPORT Function {
 public:
  template <typename... Args>
  Tensor call(Args... args);
  // void call_backward();
 protected:
  std::string name;
};

// /**
//  * @brief Compute Context
//  *
//  * Edge of the graph to connect Nodes (a.k.a. Tensors) together.
//  */
// class ABYSS_EXPORT Context {
//  public:
//   void save_for_backward(std::initializer_list<Tensor> inputs) {
//     saved_tensors_.assign(inputs.begin(), inputs.end());
//   }
//   std::vector<Tensor>& saved_tensors() { return saved_tensors_; }

//  private:
//   std::vector<Tensor> saved_tensors_;
// };

/**
 * @brief Backward function meta class
 */
class ABYSS_EXPORT BackwardFn {
 public:
  using FuncType = std::function<std::vector<Tensor>(Context&, Tensor)>;

  BackwardFn(std::string name, FuncType func) : func_(func) {
    name_.append(name);
  }

  std::vector<Tensor> call(Context ctx, Tensor output_grad) {
    return func_(ctx, output_grad);
  }

  friend std::ostream& operator<<(std::ostream& os, const BackwardFn& bkd_fn) {
    os << bkd_fn.name_ << std::endl;
    return os;
  }

 private:
  std::string name_ = "BackwardFn_";
  // Context ctx_;
  FuncType func_;
};

template <typename ChildType>
template <typename... Args>
Tensor Function<ChildType>::call(Args... args) {
  using namespace abyss::core;

  Graph& graph = Graph::instance();

  // create context and compute
  Context ctx;
  Tensor output = ChildType::forward(ctx, std::forward<Args>(args)...);

  // update properties
  std::array<Tensor, sizeof...(Args)> inputs{std::forward<Args>(args)...};
  bool requires_grad = false;
  for (auto&& in : inputs) {
    requires_grad |= in.flags(FlagId::kRequiresGrad);
  }
  output.set_flag(FlagId::kRequiresGrad, requires_grad);
  output.set_flag(FlagId::kIsLeaf, false);
  // output.set_requires_grad(true);
  // output.is_leaf_ = false;
  if (requires_grad) {
    output.grad_fn_ = std::make_shared<BackwardFn>(name, ChildType::backward);
    graph.bind_context(output, ctx);
  }
  // bind output tensor with context and register to graph
  // graph.add_node(output);
  // graph.bind_context(output, ctx);

  return output;
}

// Tensor full();

// struct AddFn : Function<AddFn> {
//   AddFn() { Function<AddFn>::name = "Add"; }
//   static Tensor forward(Context& ctx, Tensor a, Tensor b) {
//     ctx.save_for_backward({a, b});

//     return add(a, b);
//   }
//   static std::vector<Tensor> backward(Context& ctx, Tensor output_grad) {
//     auto tensors = ctx.saved_tensors();

//     tensors[0].grad() = tensors[0].grad() + output_grad;
//     tensors[1].grad() = tensors[1].grad() + output_grad;
//     std::vector<Tensor> input_grads = {tensors[0].grad(), tensors[1].grad()};

//     return input_grads;
//   }
// };

// inline ABYSS_EXPORT Tensor add(Tensor a, Tensor b) {
//   AddFn add_fn;
//   return add_fn.call(a, b);
// }

}  // namespace abyss::autograd

#endif