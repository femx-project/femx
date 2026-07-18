#pragma once

#include <femx/linalg/LinearOperator.hpp>

namespace femx
{
class DenseMatrix;

namespace linalg
{

/**
 * @brief Matrix that can be assembled and applied as a linear operator.
 *
 * MatrixOperator combines entry assembly with LinearOperator application.
 */
class MatrixOperator : public LinearOperator
{
public:
  ~MatrixOperator() override = default;

  Index numRows() const override = 0;
  Index numCols() const override = 0;

  virtual void resize(Index rows, Index cols)      = 0;
  virtual void setZero()                           = 0;
  virtual void set(Index row, Index col, Real val) = 0;
  virtual void add(Index row, Index col, Real val) = 0;
  virtual void finalize()                          = 0;

  virtual void addAtomic(Index row, Index col, Real val)
  {
    add(row, col, val);
  }

  virtual void addElem(Index               ie,
                       const Array<Index>& row_dofs,
                       const Array<Index>& col_dofs,
                       const DenseMatrix&  mat_e,
                       bool                atomic);
};

} // namespace linalg
} // namespace femx
