#pragma once

#include <vector>

#include <refem/common/Types.hpp>

namespace refem
{

class DenseMatrix
{
public:
  DenseMatrix();

  DenseMatrix(index_type rows, index_type cols);

  void resize(index_type rows, index_type cols);

  void setZero();

  index_type rows() const;
  index_type cols() const;
  index_type size() const;

  real_type& operator()(index_type i, index_type j);
  real_type  operator()(index_type i, index_type j) const;

  real_type* data();
  const real_type* data() const;

private:
  index_type rows_;
  index_type cols_;

  std::vector<real_type> values_;
};

} // namespace refem
