#include "autograd/graph.h"

#include <memory>

#include "autograd/function.h"
// #include "core/utility.h"

namespace abyss::autograd {

void Context::save_for_backward(std::initializer_list<Tensor> inputs) {
  saved_tensors_.assign(inputs.begin(), inputs.end());
}
std::vector<Tensor>& Context::saved_tensors() { return saved_tensors_; }

/**
 * Graph impementations
 */

void Graph::clear() { Graph::instance().edges_.clear(); }
Graph::EdgeType Graph::edges() const { return edges_; }

void Graph::bind_context(Tensor tsr, Context ctx) { edges_[tsr] = ctx; }

void Graph::backward(Tensor& output, Tensor output_grad) {
  // Context& ctx = edges_[output];
  auto it = edges_.find(output);
  if (it == edges_.end()) {
    // reached leaf tensor, update gradients
    if (output.flags(abyss::core::FlagId::kRequiresGrad)) {
      // output.init_grad();
      output.grad() = output.grad() + output_grad;
      
      // Tensor tmp = output.grad() + output_grad;
      // output.grad() = tmp;
    }

    return;
  }

  Context& ctx = it->second;
  auto& inputs = ctx.saved_tensors();
  // gradients are updated in the grad_fn call
  auto input_grads = output.grad_fn().call(ctx, output_grad);

  // Depth First traversal
  for (size_t i = 0; i < inputs.size(); i++) {
    backward(inputs[i], input_grads[i]);
  }
}

};  // namespace abyss::autograd