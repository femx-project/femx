#include <femx/linalg/petsc/PETScSystemMatrix.hpp>

namespace femx::linalg
{

PETScSystemMatrix::PETScSystemMatrix(MpiContext& ctx)
  : ctx_(ctx), matrix_(ctx.comm())
{
}

void PETScSystemMatrix::setup(const HostCsrPattern& pattern)
{
  matrix_.resize(pattern);
  ctx_.setPartition(matrix_.partition());
}

void PETScSystemMatrix::addElement(const ElementJacobianView& element)
{
#pragma omp critical(femx_petsc_matrix_set_value)
  {
    matrix_.addBlock(element.rows, element.columns, element.values);
  }
}

void PETScSystemMatrix::replaceRows(HostVectorView<const Index> rows,
                                    Real                        diagonal)
{
  matrix_.replaceRows(rows, diagonal);
}

void PETScSystemMatrix::eliminateColumns(
    HostVectorView<const Index> rows,
    HostVectorView<const Real>  values,
    HostVectorView<Real>        rhs)
{
  matrix_.eliminateColumns(rows, values, rhs);
}

void PETScSystemMatrix::finalize()
{
  matrix_.finalize();
}

void PETScSystemMatrix::apply(HostVectorView<const Real> direction,
                              HostVector<Real>&          out) const
{
  matrix_.apply(direction, out);
}

void PETScSystemMatrix::applyT(HostVectorView<const Real> direction,
                               HostVector<Real>&          out) const
{
  matrix_.applyT(direction, out);
}

const PETScMatrix& PETScSystemMatrix::matrix() const noexcept
{
  return matrix_;
}

} // namespace femx::linalg
