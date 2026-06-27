#include <stdexcept>

#include <femx/linalg/native/DenseMatrixOperator.hpp>

namespace femx
{
namespace linalg
{

Index DenseMatrixOperator::numRows() const
{
  return mat_.rows();
}

Index DenseMatrixOperator::numCols() const
{
  return mat_.cols();
}

void DenseMatrixOperator::resize(Index rows, Index cols)
{
  mat_.resize(rows, cols);
}

void DenseMatrixOperator::setZero()
{
  mat_.setZero();
}

void DenseMatrixOperator::set(Index row, Index col, Real value)
{
  mat_(row, col) = value;
}

void DenseMatrixOperator::add(Index row, Index col, Real value)
{
  mat_(row, col) += value;
}

void DenseMatrixOperator::addAtomic(Index row, Index col, Real value)
{
  Real& entry = mat_(row, col);
#pragma omp atomic update
  entry += value;
}

void DenseMatrixOperator::finalize()
{
}

void DenseMatrixOperator::apply(const Vector<Real>& dir,
                                Vector<Real>&       out) const
{
  if (dir.size() != numCols())
  {
    throw std::runtime_error(
        "DenseMatrixOperator apply received incompatible vector");
  }

  resizeOrZero(out, numRows());
  for (Index i = 0; i < numRows(); ++i)
  {
    for (Index j = 0; j < numCols(); ++j)
    {
      out[i] += mat_(i, j) * dir[j];
    }
  }
}

void DenseMatrixOperator::applyT(const Vector<Real>& dir,
                                 Vector<Real>&       out) const
{
  if (dir.size() != numRows())
  {
    throw std::runtime_error(
        "DenseMatrixOperator transpose apply received incompatible vector");
  }

  resizeOrZero(out, numCols());
  for (Index i = 0; i < numRows(); ++i)
  {
    for (Index j = 0; j < numCols(); ++j)
    {
      out[j] += mat_(i, j) * dir[i];
    }
  }
}

DenseMatrix& DenseMatrixOperator::mat()
{
  return mat_;
}

const DenseMatrix& DenseMatrixOperator::mat() const
{
  return mat_;
}

} // namespace linalg
} // namespace femx
