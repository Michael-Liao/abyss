#ifndef ABYSS_OPTIMIZER_H
#define ABYSS_OPTIMIZER_H

#include "abyss_export.h"
#include "tensor.h"

namespace abyss::optim {

class ABYSS_EXPORT Optimizer {
 public:
  /**
   * @brief clear gradients
   * 
   * zero_grad is neccesary because we need to clear gradient for every batch
   * so it doesn't accumulate the gradients from previous backward passes
   */
  void zero_grad();

  /**
   * @brief update model parameters based on gradients
   */
  // virtual void step() {
  //   throw std::runtime_error("pleasae implement the step function for your optimizer.");
  // }

  virtual void step() = 0;

 protected:
  std::vector<Tensor>& params_;

  /**
   * Constructor set as protected becasue this was meant to be extended.
   */
  Optimizer(std::vector<Tensor>& parameters);
};

/**
 * @brief Stochastic Gradient Descent
 *
 * no parameters and momentum and fancy stuff. Just a naive implementation.
 */
class ABYSS_EXPORT SGD final : public Optimizer {
 public:
  SGD(std::vector<Tensor>& parameters, double lr);

  void step() override;

 private:
  double learning_rate_;
};

}  // namespace abyss::optim

#endif