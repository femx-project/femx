#pragma once

#include <limits>

#include <femx/common/Checks.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

/** @brief Own row-major Host matrix storage. */
class DenseMatrix final
{
public:
  DenseMatrix() = default;

  /**
   * @brief Construct a zero-initialized matrix.
   *
   * @param[in] rows - Number of rows.
   * @param[in] cols - Number of columns.
   * @throws std::runtime_error - If either dimension is negative or their
   * product exceeds the supported size.
   */
  DenseMatrix(Index rows, Index cols)
    : rows_(rows), cols_(cols), vals_(checkedSize(rows, cols))
  {
  }

  /**
   * @brief Replace the storage with a zero-initialized matrix.
   *
   * @param[in] rows - Number of rows.
   * @param[in] cols - Number of columns.
   * @throws std::runtime_error - If either dimension is negative or their
   * product exceeds the supported size.
   */
  void resize(Index rows, Index cols)
  {
    const Index size = checkedSize(rows, cols);
    rows_            = rows;
    cols_            = cols;
    vals_.resize(size);
  }

  /** @brief Return the number of rows. */
  Index rows() const noexcept
  {
    return rows_;
  }

  /** @brief Return the number of columns. */
  Index cols() const noexcept
  {
    return cols_;
  }

  /** @brief Return the number of entries. */
  Index size() const noexcept
  {
    return vals_.size();
  }

  /** @brief Report whether the matrix has no entries. */
  bool empty() const noexcept
  {
    return vals_.empty();
  }

  /**
   * @brief Access an entry without bounds checking.
   *
   * @param[in] row - Row index.
   * @param[in] col - Column index.
   * @return Reference to the indexed entry.
   */
  Real& operator()(Index row, Index col)
  {
    return vals_[row * cols_ + col];
  }

  /**
   * @brief Access an entry without bounds checking.
   *
   * @param[in] row - Row index.
   * @param[in] col - Column index.
   * @return Read-only reference to the indexed entry.
   */
  const Real& operator()(Index row, Index col) const
  {
    return vals_[row * cols_ + col];
  }

  /** @brief Return the address of the first entry. */
  Real* data() noexcept
  {
    return vals_.data();
  }

  /** @brief Return the address of the first entry. */
  const Real* data() const noexcept
  {
    return vals_.data();
  }

  /** @brief Return a mutable row-major Host view of the matrix. */
  HostMatrixView<Real> view() noexcept
  {
    return {data(), rows_, cols_};
  }

  /** @brief Return a read-only row-major Host view of the matrix. */
  HostMatrixView<const Real> view() const noexcept
  {
    return {data(), rows_, cols_};
  }

  /** @brief Return the owned row-major entry storage. */
  HostVector& vals() noexcept
  {
    return vals_;
  }

  /** @brief Return the owned row-major entry storage. */
  const HostVector& vals() const noexcept
  {
    return vals_;
  }

private:
  static Index checkedSize(Index rows, Index cols)
  {
    require(rows >= 0 && cols >= 0,
            "DenseMatrix dimensions must be non-negative");
    if (rows != 0)
    {
      require(cols <= std::numeric_limits<Index>::max() / rows,
              "DenseMatrix dimensions exceed the supported size");
    }
    return rows * cols;
  }

  Index      rows_{0}; ///< Number of rows.
  Index      cols_{0}; ///< Number of columns.
  HostVector vals_;    ///< Row-major matrix values.
};

} // namespace femx
