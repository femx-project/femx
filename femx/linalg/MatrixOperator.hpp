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

  /** @brief Return the global row count. */
  Index numRows() const override = 0;
  /** @brief Return the global column count. */
  Index numCols() const override = 0;

  /** @brief Replace storage with a zero matrix of the requested shape. */
  virtual void resize(Index rows, Index cols)      = 0;
  /** @brief Set all matrix entries to zero. */
  virtual void setZero()                           = 0;
  /** @brief Replace one global entry. */
  virtual void set(Index row, Index col, Real val) = 0;
  /** @brief Accumulate into one global entry. */
  virtual void add(Index row, Index col, Real val) = 0;
  /** @brief Complete backend assembly before the matrix is consumed. */
  virtual void finalize()                          = 0;

  /** @brief Thread-safe entry accumulation when supported by the backend. */
  virtual void addAtomic(Index row, Index col, Real val)
  {
    add(row, col, val);
  }

  /**
   * @brief Accumulate one dense element matrix into global rows and columns.
   * @param ie Element index used by map-backed implementations.
   * @param row_dofs Global row DOFs.
   * @param col_dofs Global column DOFs.
   * @param mat_e Row-major element matrix.
   * @param atomic Request thread-safe accumulation.
   */
  virtual void addElem(Index               ie,
                       const Array<Index>& row_dofs,
                       const Array<Index>& col_dofs,
                       const DenseMatrix&  mat_e,
                       bool                atomic);
};

} // namespace linalg
} // namespace femx
