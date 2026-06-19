#pragma once

#include <femx/common/Types.hpp>

namespace femx
{
namespace system
{

/** @brief Mutable global vector formed from elem contributions. */
class SystemVector
{
public:
  virtual ~SystemVector() = default;

  virtual Index size() const = 0;

  virtual void resize(Index size)         = 0;
  virtual void setZero()                  = 0;
  virtual void set(Index row, Real value) = 0;
  virtual void add(Index row, Real value) = 0;
  virtual void finalize()                 = 0;

  virtual void addAtomic(Index row, Real value)
  {
    add(row, value);
  }
};

} // namespace system
} // namespace femx
