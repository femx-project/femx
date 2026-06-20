#pragma once

#include <femx/core/Types.hpp>

namespace femx
{

template <class T>
class MatrixView
{
public:
  MatrixView() = default;

  MatrixView(T* data, Index rows, Index cols)
    : data_(data),
      rows_(rows),
      cols_(cols)
  {
  }

  T& operator()(Index i, Index j) const
  {
    return data_[i * cols_ + j];
  }

  T* data() const
  {
    return data_;
  }

  Index rows() const
  {
    return rows_;
  }

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
