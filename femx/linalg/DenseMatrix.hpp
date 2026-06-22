#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

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

  Vector<Real> vals_;
};

} // namespace femx
