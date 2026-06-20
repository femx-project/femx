#include "Assembly.hpp"

#if defined(FEMX_HAS_PETSC)

#include <petscsys.h>

#include <stdexcept>
#include <vector>

#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/system/petsc/PETScSystemMatrix.hpp>

using namespace femx::system;

namespace femx
{

namespace
{

using ElemMatrixFn = void (*)(const MixedFESpace&,
                              Index,
                              ElementValues&,
                              const GaussQuadrature&,
                              const Vector<Real>&,
                              const Vector<Real>&,
                              const TimeNavierStokesParameters&,
                              DenseMatrix&);

void assemblePETScJac(const MixedFESpace&               space,
                      const GaussQuadrature&            quad,
                      const Vector<Real>&               x_next,
                      const Vector<Real>&               x,
                      const TimeNavierStokesParameters& prm,
                      const NavierVarCellRange&         cells,
                      PETScSystemMatrix&                out,
                      ElemMatrixFn                      elem_matrix)
{
  const Index num_dofs = space.numDofsPerElem();
  if (cells.begin < 0 || cells.end < cells.begin
      || cells.end > space.numElems())
  {
    throw std::runtime_error("PETSc Navier assembly cell range is invalid");
  }
  out.setZero();

  const auto&           elem = space.field(0).space().finiteElement();
  ElementValues         ev(elem, quad);
  DenseMatrix           Ke(num_dofs, num_dofs);
  Vector<Index>         elem_dofs;
  std::vector<PetscInt> dofs(static_cast<std::size_t>(num_dofs));

  for (Index cell = cells.begin; cell < cells.end; ++cell)
  {
    elem_matrix(space, cell, ev, quad, x_next, x, prm, Ke);

    space.elemDofs(cell, elem_dofs);
    if (elem_dofs.size() != num_dofs)
    {
      throw std::runtime_error(
          "PETSc Navier assembly element dof size mismatch");
    }
    for (Index i = 0; i < num_dofs; ++i)
    {
      dofs[static_cast<std::size_t>(i)] =
          static_cast<PetscInt>(elem_dofs[i]);
    }

    out.addBlock(dofs.data(), num_dofs, Ke);
  }
}

} // namespace

void assembleNextStateJacPETSc(const MixedFESpace&               space,
                               const GaussQuadrature&            quad,
                               const Vector<Real>&               x_next,
                               const Vector<Real>&               x,
                               const TimeNavierStokesParameters& prm,
                               const NavierVarCellRange&         cells,
                               PETScSystemMatrix&                out)
{
  assemblePETScJac(space,
                   quad,
                   x_next,
                   x,
                   prm,
                   cells,
                   out,
                   assembleNextElemMatrix);
}

void assemblePrevStateJacPETSc(const MixedFESpace&               space,
                               const GaussQuadrature&            quad,
                               const Vector<Real>&               x_next,
                               const Vector<Real>&               x,
                               const TimeNavierStokesParameters& prm,
                               const NavierVarCellRange&         cells,
                               PETScSystemMatrix&                out)
{
  assemblePETScJac(space,
                   quad,
                   x_next,
                   x,
                   prm,
                   cells,
                   out,
                   assemblePrevElemMatrix);
}

} // namespace femx

#endif
