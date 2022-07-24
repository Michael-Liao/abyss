#include "utils/data.h"

#include <algorithm>
#include <numeric>
#include <random>

#include "functional.h"

namespace abyss::utils::data {

DataLoader::DataLoader(Dataset& dataset, size_t batch_size, bool shuffle)
    : batch_size_{batch_size},
      shuffle_{shuffle},
      dataset_{&dataset},
      ids_(calc_size(dataset_->size(), batch_size_)) {
  std::iota(ids_.begin(), ids_.end(), 0);
}

DataLoader::DataLoader(const DataLoader& other)
    : batch_size_{other.batch_size_},
      shuffle_{other.shuffle_},
      dataset_{other.dataset_},
      offset_{other.offset_},
      ids_{other.ids_} {}

DataLoader& DataLoader::operator=(DataLoader copy) {
  swap(copy);
  return *this;
}
DataLoader::reference DataLoader::operator*() {
  update_slice();
  return output_slice_;
}
DataLoader::pointer DataLoader::operator->() {
  update_slice();
  return &output_slice_;
}
size_t DataLoader::size() const {
  return calc_size(dataset_->size(), batch_size_);
}

DataLoader DataLoader::begin() {
  // std::vector<int> x(10);
  if (shuffle_) {
    /// @todo reproducibility
    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::shuffle(ids_.begin(), ids_.end(), rng);
    // std::shuffle(x.begin(), x.end(), rng);
  }

  return *this;
}

DataLoader DataLoader::end() {
  DataLoader out = *this;  // copy
  out.offset_ = size();

  return out;
}

DataLoader& DataLoader::operator++() {
  offset_ += batch_size_;
  return *this;
}
DataLoader& DataLoader::operator++(int discard) { return ++*this; }

bool DataLoader::operator==(const DataLoader& other) const {
  return offset_ == other.offset_;
}
bool DataLoader::operator!=(const DataLoader& other) const {
  return offset_ != other.offset_;
}

size_t DataLoader::calc_size(size_t dataset_size, size_t batch_size) {
  return (dataset_size - 1 + batch_size) / batch_size;
}

void DataLoader::update_slice() {
  Dataset& dset = *dataset_;

  size_t real_batch_size = std::min(dset.size() - offset_, batch_size_);
  std::vector<Tensor> Xs(real_batch_size);
  std::vector<Tensor> ys(real_batch_size);

  for (size_t i = 0; i < real_batch_size; i++) {
    std::tie(Xs[i], ys[i]) = dset[ids_[offset_ +i]];
  }

  Tensor X = concat(Xs);
  Tensor y = concat(ys);

  output_slice_ = std::make_pair(X, y);
}

}  // namespace abyss::utils::data