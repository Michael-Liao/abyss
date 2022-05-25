#ifndef ABYSS_CORE_UTILITY_H
#define ABYSS_CORE_UTILITY_H

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "abyss_export.h"

namespace abyss::core {

/**
 * @brief Tensor Description class
 *
 * This struct groups properties for calculation and slicing
 */
struct ArrayDesc {
  size_t offset = 0;
  std::vector<int> shape;
  std::vector<int> strides;
};

/**
 * @brief tensor flag mask to query flags
 */
enum class FlagId : size_t {
  kIsContiguous,  // 0
  kOwnsData,      // 1
  kIsView,        // 2
  kIsEditable,    // 3
  kRequiresGrad,  // 4
  kIsLeaf         // 5
};

class TensorFlags {
 public:
  using value_t = std::underlying_type_t<FlagId>;
  using bits_t = std::bitset<sizeof(value_t)>;

  TensorFlags() {
    // default flags are contiguous and owns data
    flags_.set(0, true);
    flags_.set(1, true);
  }

  void reset() { flags_.reset(); }

  bool operator[](FlagId id) const {
    auto pos = static_cast<value_t>(id);

    return flags_[pos];
  }

  bits_t::reference operator[](FlagId id) {
    auto pos = static_cast<value_t>(id);

    return flags_[pos];
  }

 private:
  bits_t flags_;
};
// enum class ABYSS_EXPORT TensorFlags : uint64_t {
//   kNoFlags = 0,
//   kIsContiguous = 1 << 0,
//   kOwnsData = 1 << 1,
//   kIsView = 1 << 2,
//   kIsEditable = 1 << 3,
// };

// inline ABYSS_EXPORT TensorFlags operator~(TensorFlags flag) {
//   using T = std::underlying_type_t<TensorFlags>;
//   return static_cast<TensorFlags>(~static_cast<T>(flag));
// }

// inline ABYSS_EXPORT TensorFlags operator|(TensorFlags flg1, TensorFlags flg2)
// {
//   using T = std::underlying_type_t<TensorFlags>;
//   return static_cast<TensorFlags>(static_cast<T>(flg1) |
//   static_cast<T>(flg2));
// }

// inline ABYSS_EXPORT TensorFlags operator&(TensorFlags flg1, TensorFlags flg2)
// {
//   using T = std::underlying_type_t<TensorFlags>;
//   return static_cast<TensorFlags>(static_cast<T>(flg1) &
//   static_cast<T>(flg2));
// }

/**
 * @brief calculate array element size from the shape
 */
inline size_t shape2size(const std::vector<int>& shape) {
  auto length =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

  return static_cast<size_t>(length);
}

/**
 * @brief calculate strides from the shape
 */
inline std::vector<int> shape2strides(const std::vector<int>& shape) {
  std::vector<int> strides(shape.size(), 1);

  for (int i = shape.size() - 1; i >= 0; i--) {
    strides[i] = std::accumulate(shape.begin() + i + 1, shape.end(), 1,
                                 std::multiplies<int>());
  }

  return strides;
}

/**
 * @brief calcuate the coordinate/indices from a 1-d index.
 *
 * @param[in] index the 1-d index to unravel
 * @param[in] shape the shape that the coordinates is based on.
 *
 * @return the coordinates of the 1-d `index` based on the `shape`.
 */
inline std::vector<int> unravel_index(int index,
                                      const std::vector<int>& shape) {
  std::vector<int> out(shape.size());

  for (int i = shape.size() - 1; i >= 0; i--) {
    out[i] = index % shape[i];
    index /= shape[i];
  }

  if (index != 0) {
    throw std::domain_error("index out of bound for supplied shape");
  }
  // std::cout<< index << std::endl;

  return out;
}

/**
 * @brief check if the shape are boardcast compatible
 */
inline bool is_broadcastable(const std::vector<int>& shape1,
                             const std::vector<int>& shape2) noexcept {
  size_t min_dim = std::min(shape1.size(), shape2.size());

  auto it1 = shape1.rbegin();
  auto it2 = shape2.rbegin();
  for (size_t i = 0; i < min_dim; i++) {
    if (it1[i] != it2[i] && std::min(it1[i], it2[i]) != 1) return false;
  }

  return true;
}

/**
 * @brief costum copy function that takes slices into consideration
 */
template <typename InputIt, typename OutputIt>
OutputIt copy(InputIt first, InputIt last, ArrayDesc desc, OutputIt d_first,
              ArrayDesc d_desc) {
  first += desc.offset;
  d_first += d_desc.offset;

  // raw index to iterate through
  size_t index = 0;
  size_t offset = 0;
  size_t d_offset = 0;
  // size_t acc_d_offset = 0;
  while (index < shape2size(desc.shape)) {
    // calculate output offset
    auto d_coords = unravel_index(index, d_desc.shape);
    d_offset = 0;
    for (size_t i = 0; i < d_desc.strides.size(); i++) {
      d_offset += d_desc.strides[i] * d_coords[i];
    }

    // calculate input offset
    auto coords = unravel_index(index, desc.shape);
    offset = 0;
    for (size_t i = 0; i < desc.strides.size(); i++) {
      offset += desc.strides[i] * coords[i];
    }

    *(d_first + d_offset) = *(first + offset);  // assign to output

    // d_first[d_offset] = first[offset];

    // std::cout<< d_offset << ", " << offset << std::endl;
    // std::cout<< d_offset << ", " << *(first + offset) << std::endl;

    index++;
  }

  return d_first + d_offset + 1;
}

/**
 * @brief Broadcast and copy the input sequence into an output
 *
 * https://stackoverflow.com/questions/39626233/how-did-numpy-implement-multi-dimensional-broadcasting
 * TL;DR set stride of broadcast axis to 0
 *
 * @tparam InputIt Input iterator of the input sequence.
 * @tparam OutputIt Output iterator for thee destination sequence.
 * @param first The start of the input sequence.
 * @param last The end of the input sequence.
 * @param desc The description of the input.
 * @param d_first The start of the output sequence.
 * @param d_desc Target description to broadcast to.
 */
template <typename InputIt, typename OutputIt>
OutputIt broadcast_copy(InputIt first, InputIt last, ArrayDesc desc,
                        OutputIt d_first, ArrayDesc d_desc) {
  size_t extra_dim = d_desc.shape.size() - desc.shape.size();
  if (extra_dim > 0) {
    // extra dimensions require broadcast, padd zeros to the front of the
    // strides
    desc.strides.insert(desc.strides.begin(), extra_dim, 0);
  }

  // set strides of broadcastable axis to 0
  auto it = desc.shape.rbegin();
  auto strides_it = desc.strides.rbegin();
  auto d_it = d_desc.shape.rbegin();
  for (size_t i = 0; i < desc.shape.size(); i++) {
    if (*d_it / *it != 1 && *it == 1) {
      *strides_it = 0;
    }

    it++;
    strides_it++;
    d_it++;
  }

  // copy the data
  first += desc.offset;
  d_first += d_desc.offset;

  size_t arr_size = shape2size(d_desc.shape);
  // basic index to go through
  size_t index = 0;
  // actual offsets / index of the array
  size_t offset = 0;
  size_t d_offset = 0;
  while (index < arr_size) {
    auto coords = unravel_index(index, d_desc.shape);
    offset = 0;
    for (size_t i = 0; i < coords.size(); i++) {
      offset += coords[i] * desc.strides[i];
    }
    d_offset = 0;
    for (size_t i = 0; i < coords.size(); i++) {
      d_offset += coords[i] * d_desc.strides[i];
    }

    *(d_first + d_offset) = *(first + offset);

    // std::cout<< d_offset << ", " << offset<<std::endl;

    index++;
  }

  return d_first + d_offset + 1;
}

}  // namespace abyss::core

#endif