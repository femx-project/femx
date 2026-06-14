#pragma once

#include <femx/common/Types.hpp>
#include <femx/system/LinearOperator.hpp>

namespace femx
{
namespace system
{

/** @brief Mutable matrix that can also act as a LinearOperator. */
class SystemMatrix : public LinearOperator
{
public:
  ~SystemMatrix() override = default;

  virtual void resize(index_type rows, index_type cols)             = 0;
  virtual void setZero()                                            = 0;
  virtual void set(index_type row, index_type col, real_type value) = 0;
  virtual void add(index_type row, index_type col, real_type value) = 0;
  virtual void finalize()                                           = 0;

  virtual void addAtomic(index_type row, index_type col, real_type value)
  {
    add(row, col, value);
  }
};

} // namespace system
} // namespace femx
