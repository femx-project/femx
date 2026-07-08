#include <stdexcept>

#include <femx/linalg/native/CsrAssemblyMatrix.hpp>

namespace femx
{
namespace linalg
{

CsrAssemblyMatrix::CsrAssemblyMatrix(const CsrPattern& pattern)
  : mat_(pattern)
{
}

Index CsrAssemblyMatrix::numRows() const
{
  return mat_.rows();
}

Index CsrAssemblyMatrix::numCols() const
{
  return mat_.cols();
}

void CsrAssemblyMatrix::resize(Index rows, Index cols)
{
  if (rows != numRows() || cols != numCols())
  {
    throw std::runtime_error(
        "CsrAssemblyMatrix cannot resize away from its sparsity pattern");
  }
}

void CsrAssemblyMatrix::setZero()
{
  mat_.setZero();
}

void CsrAssemblyMatrix::set(Index row, Index col, Real value)
{
  mat_.valuesData()[findEntry(row, col)] = value;
}

void CsrAssemblyMatrix::add(Index row, Index col, Real value)
{
  mat_.valuesData()[findEntry(row, col)] += value;
}

void CsrAssemblyMatrix::addAtomic(Index row, Index col, Real value)
{
  Real*       vals  = mat_.valuesData();
  const Index entry = findEntry(row, col);
#pragma omp atomic update
  vals[entry] += value;
}

bool CsrAssemblyMatrix::addMappedMat(Index              ie,
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

void CsrAssemblyMatrix::finalize()
{
}

void CsrAssemblyMatrix::apply(const Vector<Real>& dir,
                              Vector<Real>&       out) const
{
  if (dir.size() != numCols())
  {
    throw std::runtime_error(
        "CsrAssemblyMatrix apply received incompatible vector");
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

void CsrAssemblyMatrix::applyT(const Vector<Real>& dir,
                               Vector<Real>&       out) const
{
  if (dir.size() != numRows())
  {
    throw std::runtime_error(
        "CsrAssemblyMatrix transpose apply received incompatible vector");
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

CsrMatrix& CsrAssemblyMatrix::mat()
{
  return mat_;
}

const CsrMatrix& CsrAssemblyMatrix::mat() const
{
  return mat_;
}

void CsrAssemblyMatrix::checkMappedMat(Index              ie,
                                       const DenseMatrix& local) const
{
  const CsrPattern& pattern = mat_.pattern();
  if (ie < 0 || ie >= pattern.numElems())
  {
    throw std::runtime_error("CsrAssemblyMatrix element index is out of range");
  }

  const Index num_dofs = pattern.elemNumDofs(ie);
  if (local.rows() != num_dofs || local.cols() != num_dofs)
  {
    throw std::runtime_error(
        "CsrAssemblyMatrix element matrix size does not match dofs");
  }
}

void CsrAssemblyMatrix::addMappedMatSerial(Index              ie,
                                           const DenseMatrix& local)
{
  checkMappedMat(ie, local);

  const CsrPattern& pattern    = mat_.pattern();
  const Index       num_dofs         = pattern.elemNumDofs(ie);
  const Index       coo_offset = pattern.elemCooOffset(ie);
  const Index*      coo_to_csr = pattern.cooToCsrData();
  const Real*       local_vals = local.data();
  Real*             vals       = mat_.valuesData();

  for (Index i = 0; i < num_dofs; ++i)
  {
    for (Index j = 0; j < num_dofs; ++j)
    {
      const Index local_entry  = i * num_dofs + j;
      const Index coo_entry    = coo_offset + local_entry;
      const Index csr_entry    = coo_to_csr[coo_entry];
      vals[csr_entry]         += local_vals[local_entry];
    }
  }
}

void CsrAssemblyMatrix::addMappedMatAtomic(Index              ie,
                                           const DenseMatrix& local)
{
  checkMappedMat(ie, local);

  const CsrPattern& pattern    = mat_.pattern();
  const Index       num_dofs         = pattern.elemNumDofs(ie);
  const Index       coo_offset = pattern.elemCooOffset(ie);
  const Index*      coo_to_csr = pattern.cooToCsrData();
  const Real*       local_vals = local.data();
  Real*             vals       = mat_.valuesData();

  for (Index i = 0; i < num_dofs; ++i)
  {
    for (Index j = 0; j < num_dofs; ++j)
    {
      const Index local_entry = i * num_dofs + j;
      const Index coo_entry   = coo_offset + local_entry;
      const Index csr_entry   = coo_to_csr[coo_entry];
#pragma omp atomic update
      vals[csr_entry] += local_vals[local_entry];
    }
  }
}

Index CsrAssemblyMatrix::findEntry(Index row, Index col) const
{
  if (row < 0 || row >= numRows() || col < 0 || col >= numCols())
  {
    throw std::runtime_error("CsrAssemblyMatrix index is out of range");
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
      "CsrAssemblyMatrix entry is outside the sparsity pattern");
}

} // namespace linalg
} // namespace femx
