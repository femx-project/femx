#pragma once

#include <limits>

#include <femx/common/Checks.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

/** @brief Owning row-major Host matrix storage. */
class DenseMatrix final
{
public:
  DenseMatrix() = default;

  DenseMatrix(Index rows, Index cols)
    : rows_(rows), cols_(cols), vals_(checkedSize(rows, cols))
  {
  }

  /** @brief Replace the storage with a zero-initialized matrix. */
  void resize(Index rows, Index cols)
  {
    const Index size = checkedSize(rows, cols);
    rows_            = rows;
    cols_            = cols;
    vals_.resize(size);
  }

  Index rows() const noexcept
  {
    return rows_;
  }

  Index cols() const noexcept
  {
    return cols_;
  }

  Index size() const noexcept
  {
    return vals_.size();
  }

  bool empty() const noexcept
  {
    return vals_.empty();
  }

  Real& operator()(Index row, Index col)
  {
    return vals_[row * cols_ + col];
  }

  const Real& operator()(Index row, Index col) const
  {
    return vals_[row * cols_ + col];
  }

  Real* data() noexcept
  {
    return vals_.data();
  }

  const Real* data() const noexcept
  {
    return vals_.data();
  }

  HostMatrixView<Real> view() noexcept
  {
    return {data(), rows_, cols_};
  }

  HostMatrixView<const Real> view() const noexcept
  {
    return {data(), rows_, cols_};
  }

  HostVector& vals() noexcept
  {
    return vals_;
  }

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

  Index      rows_{0};
  Index      cols_{0};
  HostVector vals_;
};

} // namespace femx
