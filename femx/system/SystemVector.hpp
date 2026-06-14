#pragma once

#include <femx/core/Types.hpp>

namespace femx
{
namespace system
{

/** @brief Mutable global vector assembled from elem contributions. */
class SystemVector
{
public:
  virtual ~SystemVector() = default;

  virtual index_type size() const = 0;

  virtual void resize(index_type size)              = 0;
  virtual void setZero()                            = 0;
  virtual void set(index_type row, real_type value) = 0;
  virtual void add(index_type row, real_type value) = 0;
  virtual void finalize()                           = 0;

  virtual void addAtomic(index_type row, real_type value)
  {
    add(row, value);
  }
};

} // namespace system
} // namespace femx
