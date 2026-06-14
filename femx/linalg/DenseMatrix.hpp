#pragma once

#include <vector>

#include <femx/common/Types.hpp>

namespace femx
{

class DenseMatrix
{
public:
  DenseMatrix();

  DenseMatrix(Index rows, Index cols);

  void resize(Index rows, Index cols);

  void setZero();

  Index rows() const;
  Index cols() const;
  Index size() const;

  Real& operator()(Index i, Index j);
  Real  operator()(Index i, Index j) const;

  Real*       data();
  const Real* data() const;

private:
  Index rows_;
  Index cols_;

  std::vector<Real> values_;
};

} // namespace femx
