#pragma once

#include <femx/core/Types.hpp>

namespace femx
{

template <class T>
class MatrixView
{
public:
  MatrixView() = default;

  MatrixView(T* data, index_type rows, index_type cols)
    : data_(data),
      rows_(rows),
      cols_(cols)
  {
  }

  T& operator()(index_type i, index_type j) const
  {
    return data_[i * cols_ + j];
  }

  T* data() const
  {
    return data_;
  }

  index_type rows() const
  {
    return rows_;
  }

  index_type cols() const
  {
    return cols_;
  }

private:
  T*         data_ = nullptr;
  index_type rows_ = 0;
  index_type cols_ = 0;
};

} // namespace femx
