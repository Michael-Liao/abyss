#ifndef ABYSS_AUTOGRAD_GRAPH_H
#define ABYSS_AUTOGRAD_GRAPH_H

#include <unordered_map>
#include <vector>

#include "tensor.h"

// namespace abyss {
// class Tensor;
// }

namespace abyss::autograd {
// class Tensor;

/**
 * @brief Compute Context
 *
 * Edge of the graph to connect Nodes (a.k.a. Tensors) together.
 */
class ABYSS_EXPORT Context {
 public:
  void save_for_backward(std::initializer_list<Tensor> inputs);
  std::vector<Tensor>& saved_tensors();

 private:
  std::vector<Tensor> saved_tensors_;
};
// class Context;

/**
 * @brief Computational Graph
 *
 * Aritificial Neural Networks are basically Directed Acyclic Graphs (DAG).
 * The graph is represented as adjacency lists.
 *
 * Why a singleton you say? Because I cannot think of any use of multiple
 * graphs.
 */
class ABYSS_EXPORT Graph {
 public:
  using EdgeType =
      std::unordered_map<Tensor, Context, std::hash<Tensor>, Tensor::KeyEqual>;
  ~Graph() = default;
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  static Graph& instance() {
    static Graph graph;

    return graph;
  }

  static void clear();

  // std::vector<Tensor> nodes() const { return nodes_; }
  EdgeType edges() const;

  // size_t add_node(Tensor tensor) {
  //   nodes_.emplace_back(tensor);
  //   return nodes_.size() - 1;
  // }

  /**
   * add edges
   */
  void bind_context(Tensor tsr, Context ctx);

  void backward(Tensor& output, Tensor output_grad);

 private:
  // std::vector<Tensor> nodes_;
  EdgeType edges_;

  Graph() = default;
};

}  // namespace abyss::autograd
#endif