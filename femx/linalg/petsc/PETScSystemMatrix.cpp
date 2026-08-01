#include <femx/linalg/petsc/PETScSystemMatrix.hpp>

namespace femx::linalg
{

PETScSystemMatrix::PETScSystemMatrix(MpiContext& ctx)
  : ctx_(ctx), mat_(ctx.comm())
{
}

void PETScSystemMatrix::setup(const HostCsrPattern& pattern)
{
  mat_.resize(pattern);
  ctx_.setPartition(mat_.partition());
}

void PETScSystemMatrix::addElement(const ElementJacobianView& elem)
{
#pragma omp critical(femx_petsc_matrix_set_value)
  {
    mat_.addBlock(elem.rows, elem.columns, elem.values);
  }
}

void PETScSystemMatrix::replaceRows(HostVectorView<const Index> rows,
                                    Real                        diag)
{
  mat_.replaceRows(rows, diag);
}

void PETScSystemMatrix::eliminateColumns(
    HostVectorView<const Index> rows,
    HostVectorView<const Real>  values,
    HostVectorView<Real>        rhs)
{
  mat_.eliminateColumns(rows, values, rhs);
}

void PETScSystemMatrix::finalize()
{
  mat_.finalize();
}

void PETScSystemMatrix::matvec(HostVectorView<const Real> dir,
                               HostVector<Real>&          out) const
{
  mat_.matvec(dir, out);
}

void PETScSystemMatrix::matvecT(HostVectorView<const Real> dir,
                                HostVector<Real>&          out) const
{
  mat_.matvecT(dir, out);
}

const PETScMatrix& PETScSystemMatrix::matrix() const noexcept
{
  return mat_;
}

} // namespace femx::linalg
