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

using namespace femx::assembly;
using namespace femx::system;

void assembleSystem(const MixedFESpace& space,
                    const Vector<Real>& x,
                    const Vector<Real>& xp,
                    bool                initial,
                    const FluidParams&  fluid,
                    Real                dt,
                    const CellRange&    cells,
                    PETScSystemMatrix&  A,
                    PETScSystemVector&  b,
                    AssemblyStats&      stats)
{
  const auto& elem = space.field(0).space().finiteElement();
  const auto  quad = GaussQuadrature::make(elem.referenceElement(), 2);

  SystemAssembler assembler(space);
  assembler.initMat(A);
  assembler.initVec(b);

  ElementValues        ev(elem, quad);
  std::vector<QPState> qps;
  DenseMatrix          Ke(space.numDofsPerElem(), space.numDofsPerElem());
  Vector<Real>         Fe(space.numDofsPerElem());

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

    assembler.addMat(ic, Ke, A);
    assembler.addVec(ic, Fe, b);
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
