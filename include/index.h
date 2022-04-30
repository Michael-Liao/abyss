#ifndef ABYSS_INDEX_H
#define ABYSS_INDEX_H

#include <limits>

#include "abyss_export.h"

namespace abyss {
/**
 * @brief Index class for indexing and slicing a tensor
 */
struct ABYSS_EXPORT Index {
  constexpr Index() = default;
  constexpr Index(int pos) : start{pos}, stop{pos + 1}, step{1} {}
  constexpr Index(int start, int stop, int step = 1)
      : start{start}, stop{stop}, step{step} {}

  //  private:
  const int start = 0;
  const int stop = std::numeric_limits<int>::max();
  const int step = 1;
};

/**
 * @brief flag for slicing all in a dimension
 */
static constexpr Index kAll;

}  // namespace abyss

#endif