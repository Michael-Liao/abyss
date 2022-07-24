#ifndef ABYSS_AUTOGRAD_FUNCTORS_H
#define ABYSS_AUTOGRAD_FUNCTORS_H
/**
 * Function Objects for automatic differentiation.
 *
 * Must define 2 public static functions. See below for the common function
 * signature.
 */

#include "autograd/function.h"
#include "core/dispatcher.h"
#include "functional.h"
#include "ops/matrix_ops.h"
#include "ops/vector_ops.h"
#include "tensor.h"

namespace abyss::autograd {

class AddFn : public Function<AddFn> {
 public:
  static Tensor forward(Context& ctx, Tensor a, Tensor b) {
    ctx.save_for_backward({a, b});

    // std::cout<< a << std::endl;

    using namespace abyss;
    core::DataDispatcher<Tensor> dp1(a);
    core::DataDispatcher<Tensor> dp2(b);

    core::AddVisitor add_vis(dp1.desc(), dp2.desc());
    dp1.accept(&add_vis, &dp2);

    return add_vis;
  }
  static std::vector<Tensor> backward(Context& ctx, Tensor output_grad) {
    auto& tensors = ctx.saved_tensors();

    std::vector<Tensor> input_grads(tensors.size());
    for (size_t i = 0; i < tensors.size(); i++) {
      input_grads[i] = output_grad;
    }

    return input_grads;
  }
};

class SubtractFn : public Function<SubtractFn> {
 public:
  static Tensor forward(Context& ctx, Tensor a, Tensor b) {
    ctx.save_for_backward({a, b});

    using namespace abyss;
    core::DataDispatcher<Tensor> dp1(a);
    core::DataDispatcher<Tensor> dp2(b);

    core::SubtractVisitor vis(dp1.desc(), dp2.desc());
    dp1.accept(&vis, &dp2);

    return vis;
  }
  static std::vector<Tensor> backward(Context& ctx, Tensor output_grad) {
    auto& tensors = ctx.saved_tensors();

    std::vector<Tensor> input_grads(tensors.size());
    for (size_t i = 0; i < tensors.size(); i++) {
      input_grads[i] = output_grad;
    }

    return input_grads;
  }
};

class MultiplyFn : public Function<MultiplyFn> {
 public:
  static Tensor forward(Context& ctx, Tensor a, Tensor b) {
    ctx.save_for_backward({a, b});

    using namespace abyss;
    core::DataDispatcher<Tensor> dp1(a);
    core::DataDispatcher<Tensor> dp2(b);

    core::MultiplyVisitor vis(dp1.desc(), dp2.desc());
    dp1.accept(&vis, &dp2);

    return vis;
  }
  static std::vector<Tensor> backward(Context& ctx, Tensor output_grad) {
    auto& tensors = ctx.saved_tensors();

    std::vector<Tensor> input_grads(tensors.size());
    for (size_t i = 0; i < tensors.size(); i++) {
      input_grads[i] = output_grad;
    }

    return input_grads;
  }
};

class DivideFn : public Function<DivideFn> {
 public:
  static Tensor forward(Context& ctx, Tensor a, Tensor b) {
    ctx.save_for_backward({a, b});

    using namespace abyss;
    core::DataDispatcher<Tensor> dp1(a);
    core::DataDispatcher<Tensor> dp2(b);

    core::DivideVisitor vis(dp1.desc(), dp2.desc());
    dp1.accept(&vis, &dp2);

    return vis;
  }
  static std::vector<Tensor> backward(Context& ctx, Tensor output_grad) {
    auto& tensors = ctx.saved_tensors();

    std::vector<Tensor> input_grads(tensors.size());
    for (size_t i = 0; i < tensors.size(); i++) {
      input_grads[i] = output_grad;
    }

    return input_grads;
  }
};

class MatmulFn : public Function<MatmulFn> {
 public:
  static Tensor forward(Context& ctx, Tensor a, Tensor b) {
    ctx.save_for_backward({a, b});

    // std::cout<< a << std::endl;

    using namespace abyss;
    core::DataDispatcher<Tensor> dp1(a);
    core::DataDispatcher<Tensor> dp2(b);

    core::MatmulVisitor vis(dp1.desc(), dp2.desc());
    dp1.accept(&vis, &dp2);

    return vis;
  }
  static std::vector<Tensor> backward(Context& ctx, Tensor output_grad) {
    auto& inputs = ctx.saved_tensors();

    using namespace abyss;
    std::vector<Tensor> input_grads(inputs.size());

    core::DataDispatcher<Tensor> o_grad = output_grad.T();
    core::DataDispatcher<Tensor> a = inputs[0];
    core::DataDispatcher<Tensor> b = inputs[1].T();

    core::MatmulVisitor vis(o_grad.desc(), a.desc());
    o_grad.accept(&vis, &a);
    // a.accept(&vis, &o_grad);
    input_grads[1] = (vis.T().copy());

    // std::cout<< input_grads[1] <<std::endl;
    // std::cout<< inputs[0] <<std::endl;

    o_grad = output_grad;
    vis = core::MatmulVisitor(o_grad.desc(), b.desc());
    // o_grad.accept(&vis, &a);
    o_grad.accept(&vis, &b);
    input_grads[0] = vis.copy();
    // for (size_t i = 0; i < inputs.size(); i++) {

    //   /// NOTE: matmul might have mismatch because we need transpose
    //   core::DataDispatcher<Tensor> dp2(inputs[inputs.size() - i - 1]);
    //   core::MatmulVisitor vis(dp1.desc(), dp2.desc());
    //   dp1.accept(&vis, &dp2);

    //   input_grads[i] = vis;
    // }

    return input_grads;
  }
};

class ExpFn : public Function<ExpFn> {
 public:
  static Tensor forward(Context& ctx, Tensor a) {
    ctx.save_for_backward({a});

    core::DataDispatcher<Tensor> dp = a;
    core::ExpVisitor vis(dp.desc());

    dp.accept(&vis);

    return vis;
  }
  static std::vector<Tensor> backward(Context& ctx, Tensor output_grad) {
    auto inputs = ctx.saved_tensors();

    core::DataDispatcher<Tensor> dp = inputs[0];
    core::ExpVisitor vis(dp.desc());

    dp.accept(&vis);

    return {output_grad * vis};
  }
};

class SumFn : public Function<SumFn> {
 public:
  static Tensor forward(Context& ctx, Tensor a,
                        int axis = core::ReductionVisitor::kNoAxis) {
    ctx.save_for_backward({a});

    core::DataDispatcher<Tensor> dp = a;
    core::SumVisitor vis(dp.desc(), int(axis));

    dp.accept(&vis);

    return vis;
  }

  static std::vector<Tensor> backward(Context& ctx, Tensor output_grad) {
    auto inputs = ctx.saved_tensors();

    auto input_grad = output_grad.broadcast_to(inputs[0].shape());

    return {input_grad};
  }
};

class NegateFn : public Function<NegateFn> {
 public:
  static Tensor forward(Context& ctx, Tensor a) {
    ctx.save_for_backward({a});

    core::DataDispatcher<Tensor> dp = a;
    core::NegateVisitor vis(dp.desc());

    dp.accept(&vis);

    return vis;
  }
  static std::vector<Tensor> backward(Context& ctx, Tensor output_grad) {
    return {output_grad};
  }
};

class LogFn : public Function<LogFn> {
 public:
  static Tensor forward(Context& ctx, Tensor a) {
    ctx.save_for_backward({a});

    core::DataDispatcher<Tensor> dp = a;
    core::LogVisitor vis(dp.desc());

    dp.accept(&vis);

    return vis;
  }
  static std::vector<Tensor> backward(Context& ctx, Tensor output_grad) {
    auto inputs = ctx.saved_tensors();

    return {output_grad * divide(1, inputs[0])};
  }
};

}  // namespace abyss::autograd
#endif