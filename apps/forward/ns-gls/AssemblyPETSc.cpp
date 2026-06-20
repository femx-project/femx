#include <petscsys.h>

#include <stdexcept>
#include <vector>

#include "Assembly.hpp"
#include "Components.hpp"
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/system/petsc/PETScSystemMatrix.hpp>
#include <femx/system/petsc/PETScSystemVector.hpp>

namespace femx
{

using namespace femx::system;

void assembleSystem(const MixedFESpace& space,
                    const Vector<Real>& x,
                    const Vector<Real>& xp,
                    bool                initial,
                    const FluidParams&  fluid,
                    Real                dt,
                    const IndexSetList& elem_dofs,
                    const CellRange&    cells,
                    PETScSystemMatrix&  A,
                    PETScSystemVector&  b,
                    AssemblyStats&      stats)
{
  if (elem_dofs.numSets() != space.numElems())
  {
    throw std::runtime_error(
        "PETSc assembly element dof cache has incompatible size");
  }

  const auto& elem = space.field(0).space().finiteElement();
  const auto  quad = GaussQuadrature::make(elem.referenceElement(), 2);

  A.setZero();
  b.setZero();

  ElementValues         ev(elem, quad);
  std::vector<QPState>  qps;
  DenseMatrix           Ke(space.numDofsPerElem(), space.numDofsPerElem());
  Vector<Real>          Fe(space.numDofsPerElem());
  std::vector<PetscInt> dofs(static_cast<std::size_t>(space.numDofsPerElem()));

  Real cfl_local = 0.0;

  for (Index ic = cells.begin; ic < cells.end; ++ic)
  {
    assembleElemSystem(space,
                       ic,
                       ev,
                       qps,
                       x,
                       xp,
                       initial,
                       fluid,
                       dt,
                       Ke,
                       Fe,
                       cfl_local);

    const Vector<Index> elem_dof = elem_dofs.set(ic);
    if (elem_dof.size() != space.numDofsPerElem())
    {
      throw std::runtime_error(
          "PETSc assembly element dof cache has incompatible cell size");
    }
    for (Index i = 0; i < elem_dof.size(); ++i)
    {
      dofs[static_cast<std::size_t>(i)] =
          static_cast<PetscInt>(elem_dof[i]);
    }

    A.addBlock(dofs.data(), elem_dof.size(), Ke);
    b.addValues(dofs.data(), elem_dof.size(), Fe);
  }

  A.finalize();
  b.finalize();

  Real cfl_global = 0.0;
  MPI_Allreduce(&cfl_local,
                &cfl_global,
                1,
                MPIU_REAL,
                MPI_MAX,
                A.comm());
  stats.max_cfl = cfl_global;
}

} // namespace femx
