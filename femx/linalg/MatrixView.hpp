#pragma once

#include <femx/common/Types.hpp>

namespace femx
{

/** @brief Non-owning row-major view of a dense host matrix. */
template <class T>
class MatrixView
{
public:
  MatrixView() = default;

  /** @brief View a contiguous `rows` by `cols` row-major matrix. */
  MatrixView(T* data, Index rows, Index cols)
    : data_(data),
      rows_(rows),
      cols_(cols)
  {
  }

  /** @brief Access entry `(i, j)` without bounds checking. */
  T& operator()(Index i, Index j) const
  {
    return data_[i * cols_ + j];
  }

  /** @brief Return the first viewed address. */
  T* data() const
  {
    return data_;
  }

  /** @brief Return the row count. */
  Index rows() const
  {
    return rows_;
  }

  /** @brief Return the column count. */
  Index cols() const
  {
    return cols_;
  }

private:
  T*    data_ = nullptr;
  Index rows_ = 0;
  Index cols_ = 0;
};

} // namespace femx
