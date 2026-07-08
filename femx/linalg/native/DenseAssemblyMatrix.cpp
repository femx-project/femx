#include <stdexcept>

#include <femx/linalg/native/DenseAssemblyMatrix.hpp>

namespace femx
{
namespace linalg
{

Index DenseAssemblyMatrix::numRows() const
{
  return mat_.rows();
}

Index DenseAssemblyMatrix::numCols() const
{
  return mat_.cols();
}

void DenseAssemblyMatrix::resize(Index rows, Index cols)
{
  mat_.resize(rows, cols);
}

void DenseAssemblyMatrix::setZero()
{
  mat_.setZero();
}

void DenseAssemblyMatrix::set(Index row, Index col, Real value)
{
  mat_(row, col) = value;
}

void DenseAssemblyMatrix::add(Index row, Index col, Real value)
{
  mat_(row, col) += value;
}

void DenseAssemblyMatrix::addAtomic(Index row, Index col, Real value)
{
  Real& entry = mat_(row, col);
#pragma omp atomic update
  entry += value;
}

void DenseAssemblyMatrix::finalize()
{
}

void DenseAssemblyMatrix::apply(const Vector<Real>& dir,
                                Vector<Real>&       out) const
{
  if (dir.size() != numCols())
  {
    throw std::runtime_error(
        "DenseAssemblyMatrix apply received incompatible vector");
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

void DenseAssemblyMatrix::applyT(const Vector<Real>& dir,
                                 Vector<Real>&       out) const
{
  if (dir.size() != numRows())
  {
    throw std::runtime_error(
        "DenseAssemblyMatrix transpose apply received incompatible vector");
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

DenseMatrix& DenseAssemblyMatrix::mat()
{
  return mat_;
}

const DenseMatrix& DenseAssemblyMatrix::mat() const
{
  return mat_;
}

} // namespace linalg
} // namespace femx
