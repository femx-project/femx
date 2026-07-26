#include <femx/linalg/petsc/PETScJacobian.hpp>

namespace femx::linalg
{

PETScJacobian::PETScJacobian(MpiContext& ctx)
  : ctx_(ctx), matrix_(ctx.comm())
{
}

void PETScJacobian::begin(const HostCsrPattern& pattern)
{
  matrix_.resize(pattern);
}

void PETScJacobian::addElement(const ElementJacobianView& element)
{
#pragma omp critical(femx_petsc_matrix_set_value)
  {
    matrix_.addBlock(element.rows, element.columns, element.values);
  }
}

void PETScJacobian::replaceRows(HostVectorView<const Index> rows,
                                Real                        diagonal)
{
  matrix_.replaceRows(rows, diagonal);
}

void PETScJacobian::eliminateColumns(
    HostVectorView<const Index> rows,
    HostVectorView<const Real>  values,
    HostVectorView<Real>        rhs)
{
  matrix_.eliminateColumns(rows, values, rhs);
}

void PETScJacobian::finalize()
{
  matrix_.finalize();
}

void PETScJacobian::apply(HostVectorView<const Real> direction,
                          HostVector<Real>&          out) const
{
  matrix_.apply(direction, out);
}

void PETScJacobian::applyT(HostVectorView<const Real> direction,
                           HostVector<Real>&          out) const
{
  matrix_.applyT(direction, out);
}

const PETScMatrix& PETScJacobian::matrix() const noexcept
{
  return matrix_;
}

} // namespace femx::linalg
