#pragma once

#include <femx/common/Types.hpp>
#include <femx/system/LinearOperator.hpp>

namespace femx
{

class DenseMatrix;

namespace system
{

/** @brief Mutable matrix that can also act as a LinearOperator. */
class SystemMatrix : public LinearOperator
{
public:
  ~SystemMatrix() override = default;

  virtual void resize(Index rows, Index cols)        = 0;
  virtual void setZero()                             = 0;
  virtual void set(Index row, Index col, Real value) = 0;
  virtual void add(Index row, Index col, Real value) = 0;
  virtual void finalize()                            = 0;

  virtual bool addLocalMatrix(Index, const DenseMatrix&, bool)
  {
    return false;
  }

  virtual void addAtomic(Index row, Index col, Real value)
  {
    add(row, col, value);
  }
};

} // namespace system
} // namespace femx
