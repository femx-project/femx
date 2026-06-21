#pragma once

#include <femx/common/Types.hpp>

namespace femx
{

class DenseMatrix;

namespace linalg
{

/** @brief Mutable assembly target for matrix entries. */
class MatrixBuilder
{
public:
  virtual ~MatrixBuilder() = default;

  virtual Index numRows() const = 0;
  virtual Index numCols() const = 0;

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

} // namespace linalg
} // namespace femx
