#include <stdexcept>

#include <femx/linalg/native/CsrMatrixOperator.hpp>

namespace femx
{
namespace linalg
{

CsrMatrixOperator::CsrMatrixOperator(const CsrPattern& pettern)
  : mat_(pettern)
{
}

Index CsrMatrixOperator::numRows() const
{
  return mat_.rows();
}

Index CsrMatrixOperator::numCols() const
{
  return mat_.cols();
}

void CsrMatrixOperator::resize(Index rows, Index cols)
{
  if (rows != numRows() || cols != numCols())
  {
    throw std::runtime_error(
        "CsrMatrixOperator cannot resize away from its sparsity pattern");
  }
}

void CsrMatrixOperator::setZero()
{
  mat_.setZero();
}

void CsrMatrixOperator::set(Index row, Index col, Real value)
{
  mat_.valuesData()[findEntry(row, col)] = value;
}

void CsrMatrixOperator::add(Index row, Index col, Real value)
{
  mat_.valuesData()[findEntry(row, col)] += value;
}

void CsrMatrixOperator::addAtomic(Index row, Index col, Real value)
{
  Real*       vals  = mat_.valuesData();
  const Index entry = findEntry(row, col);
#pragma omp atomic update
  vals[entry] += value;
}

bool CsrMatrixOperator::addMappedMat(Index              ie,
                                     const DenseMatrix& local,
                                     bool               atomic)
{
  if (atomic)
  {
    addMappedMatAtomic(ie, local);
  }
  else
  {
    addMappedMatSerial(ie, local);
  }
  return true;
}

void CsrMatrixOperator::finalize()
{
}

void CsrMatrixOperator::apply(const Vector<Real>& dir,
                              Vector<Real>&       out) const
{
  if (dir.size() != numCols())
  {
    throw std::runtime_error(
        "CsrMatrixOperator apply received incompatible vector");
  }

  resizeOrZero(out, numRows());
  const Index* rp   = mat_.rowPtrData();
  const Index* ci   = mat_.colIndData();
  const Real*  vals = mat_.valuesData();

  for (Index row = 0; row < numRows(); ++row)
  {
    Real sum = 0.0;
    for (Index k = rp[row]; k < rp[row + 1]; ++k)
    {
      sum += vals[k] * dir[ci[k]];
    }
    out[row] = sum;
  }
}

void CsrMatrixOperator::applyT(const Vector<Real>& dir,
                               Vector<Real>&       out) const
{
  if (dir.size() != numRows())
  {
    throw std::runtime_error(
        "CsrMatrixOperator transpose apply received incompatible vector");
  }

  resizeOrZero(out, numCols());
  const Index* rp   = mat_.rowPtrData();
  const Index* ci   = mat_.colIndData();
  const Real*  vals = mat_.valuesData();

  for (Index row = 0; row < numRows(); ++row)
  {
    for (Index k = rp[row]; k < rp[row + 1]; ++k)
    {
      out[ci[k]] += vals[k] * dir[row];
    }
  }
}

CsrMatrix& CsrMatrixOperator::mat()
{
  return mat_;
}

const CsrMatrix& CsrMatrixOperator::mat() const
{
  return mat_;
}

void CsrMatrixOperator::checkMappedMat(Index              ie,
                                       const DenseMatrix& local) const
{
  const CsrPattern& pettern = mat_.pettern();
  if (ie < 0 || ie >= pettern.numElems())
  {
    throw std::runtime_error("CsrMatrixOperator element index is out of range");
  }

  const Index nd = pettern.elemNumDofs(ie);
  if (local.rows() != nd || local.cols() != nd)
  {
    throw std::runtime_error(
        "CsrMatrixOperator element matrix size does not match dofs");
  }
}

void CsrMatrixOperator::addMappedMatSerial(Index              ie,
                                           const DenseMatrix& local)
{
  checkMappedMat(ie, local);

  const CsrPattern& pettern    = mat_.pettern();
  const Index       nd         = pettern.elemNumDofs(ie);
  const Index       coo_offset = pettern.elemCooOffset(ie);
  const Index*      coo_to_csr = pettern.cooToCsrData();
  const Real*       local_vals = local.data();
  Real*             vals       = mat_.valuesData();

  for (Index i = 0; i < nd; ++i)
  {
    for (Index j = 0; j < nd; ++j)
    {
      const Index local_entry  = i * nd + j;
      const Index coo_entry    = coo_offset + local_entry;
      const Index csr_entry    = coo_to_csr[coo_entry];
      vals[csr_entry]         += local_vals[local_entry];
    }
  }
}

void CsrMatrixOperator::addMappedMatAtomic(Index              ie,
                                           const DenseMatrix& local)
{
  checkMappedMat(ie, local);

  const CsrPattern& pettern    = mat_.pettern();
  const Index       nd         = pettern.elemNumDofs(ie);
  const Index       coo_offset = pettern.elemCooOffset(ie);
  const Index*      coo_to_csr = pettern.cooToCsrData();
  const Real*       local_vals = local.data();
  Real*             vals       = mat_.valuesData();

  for (Index i = 0; i < nd; ++i)
  {
    for (Index j = 0; j < nd; ++j)
    {
      const Index local_entry = i * nd + j;
      const Index coo_entry   = coo_offset + local_entry;
      const Index csr_entry   = coo_to_csr[coo_entry];
#pragma omp atomic update
      vals[csr_entry] += local_vals[local_entry];
    }
  }
}

Index CsrMatrixOperator::findEntry(Index row, Index col) const
{
  if (row < 0 || row >= numRows() || col < 0 || col >= numCols())
  {
    throw std::runtime_error("CsrMatrixOperator index is out of range");
  }

  const Index* rp = mat_.rowPtrData();
  const Index* ci = mat_.colIndData();
  for (Index k = rp[row]; k < rp[row + 1]; ++k)
  {
    if (ci[k] == col)
    {
      return k;
    }
  }

  throw std::runtime_error(
      "CsrMatrixOperator entry is outside the sparsity pattern");
}

} // namespace linalg
} // namespace femx
