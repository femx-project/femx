#include <petscsys.h>

#include <vector>

#include "Assembly.hpp"
#include "Components.hpp"
#include <femx/assembly/SystemAssembler.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FiniteElement.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/system/petsc/PETScSystemMatrix.hpp>
#include <femx/system/petsc/PETScSystemVector.hpp>

namespace femx
{

void assembleSystem(const MixedFESpace&        space,
                    const Vector&              x,
                    const Vector&              xp,
                    bool                       initial,
                    const FluidParams&         fluid,
                    real_type                  dt,
                    const CellRange&           cells,
                    system::PETScSystemMatrix& A,
                    system::PETScSystemVector& b,
                    AssemblyStats&             stats)
{
  const auto& elem = space.field(0).space().finiteElement();
  const auto  quad = GaussQuadrature::make(elem.referenceElement(), 2);
  const auto  nq   = quad.size();

  assembly::SystemAssembler assembler(space);
  assembler.initMat(A);
  assembler.initVec(b);

  ElementValues        ev(elem, quad);
  std::vector<QPState> qps(static_cast<std::size_t>(nq));
  DenseMatrix          Ke(space.numDofsPerElem(), space.numDofsPerElem());
  Vector               Fe(space.numDofsPerElem());

  real_type cfl_local = 0.0;

  for (index_type ic = cells.begin; ic < cells.end; ++ic)
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

    assembler.addMat(ic, Ke, A);
    assembler.addVec(ic, Fe, b);
  }

  A.finalize();
  b.finalize();

  real_type cfl_global = 0.0;
  MPI_Allreduce(&cfl_local,
                &cfl_global,
                1,
                MPIU_REAL,
                MPI_MAX,
                A.comm());
  stats.max_cfl = cfl_global;
}

} // namespace femx
