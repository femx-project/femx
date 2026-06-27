#pragma once

#include <algorithm>
#include <stdexcept>

#include <femx/common/Types.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace femx::runtime
{

struct IndexRange
{
  Index begin = 0;
  Index end   = 0;
};

inline void setSerialOpenMp()
{
#if defined(_OPENMP)
  omp_set_dynamic(0);
  omp_set_num_threads(1);
#endif
}

inline IndexRange partitionRange(Index count,
                                 Index rank,
                                 Index size)
{
  if (count < 0)
  {
    throw std::runtime_error("partitionRange requires nonnegative count");
  }
  if (rank < 0 || size <= 0 || rank >= size)
  {
    throw std::runtime_error("partitionRange received invalid rank/size");
  }

  const Index base  = count / size;
  const Index extra = count % size;
  const Index begin = rank * base + std::min(rank, extra);
  const Index width = base + (rank < extra ? 1 : 0);
  return {begin, begin + width};
}

} // namespace femx::runtime
