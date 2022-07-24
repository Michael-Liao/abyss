#ifndef ABYSS_UTIL_DATA_H
#define ABYSS_UTIL_DATA_H

#include <iterator>
#include <utility>
#include <memory>

#include "abyss_export.h"
#include "index.h"
#include "tensor.h"

namespace abyss::utils::data {

/**
 * @brief dataset that groups X, y together
 */
class ABYSS_EXPORT Dataset {
 public:
  virtual size_t size() const = 0;
  virtual std::pair<Tensor, Tensor> operator[](size_t idx) = 0;
};

/**
 * @brief Data loader that loads data into batches and preprocessing
 *
 * DataLoader is a iterator itself that generates slices from the dataset
 */
class ABYSS_EXPORT DataLoader {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = std::pair<Tensor, Tensor>;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::input_iterator_tag;

  DataLoader(Dataset& dataset, size_t batch_size = 1, bool shuffle = false);

  DataLoader(const DataLoader& other);
  DataLoader& operator=(DataLoader copy);

  void swap(DataLoader& b) {
    using std::swap;

    swap(batch_size_, b.batch_size_);
    swap(shuffle_, b.shuffle_);

    swap(dataset_, b.dataset_);
    swap(offset_, b.offset_);
    swap(ids_, b.ids_);
  }

  reference operator*();
  pointer operator->();

  size_t size() const;

  /**
   * non-const because we need to shuffle before iteration
   */
  DataLoader begin();
  DataLoader end();

  DataLoader& operator++();
  DataLoader& operator++(int discard);

  bool operator==(const DataLoader& other) const;
  bool operator!=(const DataLoader& other) const;

 private:
  size_t batch_size_;
  bool shuffle_;

  Dataset* dataset_;
  
  size_t offset_ = 0;
  // std::shared_ptr<size_t[]> ids_;
  std::vector<size_t> ids_;
  
  // like the view object in tensor, created for reference
  std::pair<Tensor, Tensor> output_slice_;

  static size_t calc_size(size_t dataset_size, size_t batch_size);

  void update_slice();
};

}  // namespace abyss::utils::data

#endif