#ifndef ABYSS_NN_MODULE_H
#define ABYSS_NN_MODULE_H

// #include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "abyss_export.h"
#include "autograd/graph.h"
#include "functional.h"
#include "tensor.h"

/**
 * @brief initializes parameter
 */
#define ABYSS_REGISTER_PARAM(name)     \
  abyss::nn::Module::parameters_.emplace_back(name) \
  this->local_states_[#name] = &abyss::nn::Module::parameters_.back();

#define INIT_MODULE(name, module_name, ...) \
  name = abyss::nn::module_name(__VA_ARGS__); \
  for (auto &&s : name.local_states_) { \
    std::string param_name = #name + "." + s.first(); \
    Module::states_[param_name]; \
  }
  

namespace abyss::nn {

ABYSS_EXPORT Tensor& make_parameter(Tensor data, bool requires_grad = true);

/**
 * @brief meta module class which everyone should inherit from.
 *
 * Use reflection to record parameter names? (state_dict)
 */
class ABYSS_EXPORT Module {
 public:
  virtual ~Module() = default;

  std::vector<Tensor>& parameters();

  template <typename... Args>
  Tensor operator()(Args... args) {
    return forward(std::forward<Tensor>(args)...);
  }

 protected:
  static std::vector<Tensor> parameters_;
  static std::unordered_map<std::string, Tensor*> states_;

  std::unordered_map<std::string, Tensor*> local_states_;

  Module() = default;
  // forward functions are no-op because different child might have different
  // signatures for normal layers
  virtual Tensor forward(Tensor input) { return {}; }
  // for losses
  virtual Tensor forward(Tensor input, Tensor target) { return {}; }

  friend Tensor& make_parameter(Tensor data, bool requires_grad);
};

class ABYSS_EXPORT Linear : public Module {
 public:
  Linear(int in_features, int out_features, bool bias = true);

  Tensor weight() const { return weight_; }
  Tensor bias() const { return bias_; }

  Tensor forward(Tensor input) override;

 private:
  Tensor weight_;
  Tensor bias_;
};

}  // namespace abyss::nn

#endif