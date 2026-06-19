#include "NavierStokesEquation.hpp"

#include <stdexcept>

#include "Assembly.hpp"
#include <femx/assembly/SystemAssembler.hpp>
#include <femx/fem/ElementValues.hpp>
#if defined(FEMX_HAS_PETSC)
#include <petscsys.h>

#include <femx/system/petsc/PETScSystemMatrix.hpp>
#endif

using namespace femx::assembly;
using namespace femx::system;

namespace femx
{

namespace
{

#if defined(FEMX_HAS_PETSC)
void allreduceResidual(Vector<Real>& out)
{
  const int ierr = MPI_Allreduce(MPI_IN_PLACE,
                                 out.data(),
                                 static_cast<int>(out.size()),
                                 MPIU_REAL,
                                 MPI_SUM,
                                 PETSC_COMM_WORLD);
  if (ierr != MPI_SUCCESS)
  {
    throw std::runtime_error(
        "NavierStokesEquation MPI_Allreduce failed");
  }
}
#endif

} // namespace

NavierStokesEquation::NavierStokesEquation(
    const MixedFESpace&        space,
    TimeNavierStokesParameters parameters)
  : space_(space),
    prm_(parameters),
    quad_(GaussQuadrature::make(
        space.field(0).space().finiteElement().referenceElement(),
        parameters.quad_order)),
    cell_end_(space.numElems())
{
  checkSpace();
  checkParameters();
}

void NavierStokesEquation::setCellRange(Index begin,
                                                    Index end)
{
  if (begin < 0 || end < begin || end > space_.numElems())
  {
    throw std::runtime_error(
        "NavierStokesEquation received invalid cell range");
  }
#if !defined(FEMX_HAS_PETSC)
  if (begin != 0 || end != space_.numElems())
  {
    throw std::runtime_error(
        "NavierStokesEquation cell ranges require PETSc");
  }
#endif
  cell_begin_ = begin;
  cell_end_   = end;
}

Index NavierStokesEquation::numSteps() const
{
  return prm_.steps;
}

Index NavierStokesEquation::numStates() const
{
  return space_.numDofs();
}

Index NavierStokesEquation::numParams() const
{
  return 0;
}

Index NavierStokesEquation::numRes() const
{
  return space_.numDofs();
}

void NavierStokesEquation::res(Index               step,
                                           const Vector<Real>& x_next,
                                           const Vector<Real>& x,
                                           const Vector<Real>& prm,
                                           Vector<Real>&       out) const
{
  checkSizes(step, x_next, x, prm);

  const auto&               elem = space_.field(0).space().finiteElement();
  SystemAssembler initializer(space_);
  initializer.initVec(out);

#pragma omp parallel
  {
    ElementValues             ev(elem, quad_);
    Vector<Real>              Re(space_.numDofsPerElem());
    SystemAssembler assembler(space_, AssemblyMode::Atomic);

#pragma omp for
    for (Index ic = cell_begin_; ic < cell_end_; ++ic)
    {
      assembleElemResidual(space_, ic, ev, x_next, x, prm_, Re);
      assembler.addVec(ic, Re, out);
    }
  }

#if defined(FEMX_HAS_PETSC)
  if (cell_begin_ != 0 || cell_end_ != space_.numElems())
  {
    allreduceResidual(out);
  }
#endif
}

void NavierStokesEquation::assembleNextStateJac(
    Index                 step,
    const Vector<Real>&   x_next,
    const Vector<Real>&   x,
    const Vector<Real>&   prm,
    SystemMatrix& out) const
{
  checkSizes(step, x_next, x, prm);

#if !defined(FEMX_HAS_ENZYME)
  throw std::runtime_error(
      "Next-state Jacobian requires FEMX_ENABLE_ENZYME=ON");
#endif

#if defined(FEMX_HAS_PETSC)
  if (auto* petsc = dynamic_cast<PETScSystemMatrix*>(&out))
  {
    assembleNextStateJacPETSc(
        space_, quad_, x_next, x, prm_, {cell_begin_, cell_end_}, *petsc);
    return;
  }
#endif

  SystemAssembler initializer(space_);
  initializer.initMat(out);

  const auto& elem = space_.field(0).space().finiteElement();

#pragma omp parallel
  {
    ElementValues             ev(elem, quad_);
    DenseMatrix               Ke(space_.numDofsPerElem(), space_.numDofsPerElem());
    SystemAssembler assembler(space_, AssemblyMode::Atomic);

#pragma omp for
    for (Index ic = 0; ic < space_.numElems(); ++ic)
    {
      assembleNextElemMatrix(space_, ic, ev, quad_, x_next, x, prm_, Ke);
      assembler.addMat(ic, Ke, out);
    }
  }
}

void NavierStokesEquation::assemblePrevStateJac(
    Index                 step,
    const Vector<Real>&   x_next,
    const Vector<Real>&   x,
    const Vector<Real>&   prm,
    SystemMatrix& out) const
{
  checkSizes(step, x_next, x, prm);

#if !defined(FEMX_HAS_ENZYME)
  throw std::runtime_error(
      "Previous-state Jacobian requires FEMX_ENABLE_ENZYME=ON");
#endif

#if defined(FEMX_HAS_PETSC)
  if (auto* petsc = dynamic_cast<PETScSystemMatrix*>(&out))
  {
    assemblePrevStateJacPETSc(
        space_, quad_, x_next, x, prm_, {cell_begin_, cell_end_}, *petsc);
    return;
  }
#endif

  SystemAssembler initializer(space_);
  initializer.initMat(out);

  const auto& elem = space_.field(0).space().finiteElement();

#pragma omp parallel
  {
    ElementValues             ev(elem, quad_);
    DenseMatrix               Ke(space_.numDofsPerElem(), space_.numDofsPerElem());
    SystemAssembler assembler(space_, AssemblyMode::Atomic);

#pragma omp for
    for (Index ic = 0; ic < space_.numElems(); ++ic)
    {
      assemblePrevElemMatrix(space_, ic, ev, quad_, x_next, x, prm_, Ke);
      assembler.addMat(ic, Ke, out);
    }
  }
}

void NavierStokesEquation::assembleParamJac(
    Index                 step,
    const Vector<Real>&   x_next,
    const Vector<Real>&   x,
    const Vector<Real>&   prm,
    SystemMatrix& out) const
{
  checkSizes(step, x_next, x, prm);
  out.resize(numRes(), numParams());
  out.setZero();
}

void NavierStokesEquation::checkSizes(
    Index               step,
    const Vector<Real>& x_next,
    const Vector<Real>& x,
    const Vector<Real>& prm) const
{
  if (step < 0 || step >= numSteps())
  {
    throw std::runtime_error("NavierStokesEquation step is out of range");
  }
  if (x_next.size() != numStates()
      || x.size() != numStates()
      || prm.size() != numParams())
  {
    throw std::runtime_error("NavierStokesEquation size mismatch");
  }
}

void NavierStokesEquation::checkSpace() const
{
  if (space_.numFields() != 2)
  {
    throw std::runtime_error(
        "NavierStokesEquation requires velocity and pressure fields");
  }

  const auto  u_dof = space_.field(0);
  const auto  p_dof = space_.field(1);
  const Index nd    = u_dof.space().finiteElement().dim();

  if (u_dof.numComponents() != nd)
  {
    throw std::runtime_error(
        "NavierStokesEquation velocity components must match mesh dimension");
  }
  if (p_dof.numComponents() != 1)
  {
    throw std::runtime_error(
        "NavierStokesEquation pressure field must be scalar");
  }
  if (p_dof.space().finiteElement().referenceElement()
      != u_dof.space().finiteElement().referenceElement())
  {
    throw std::runtime_error(
        "NavierStokesEquation fields must use the same reference cell");
  }
  if (p_dof.numShapesPerElem() != u_dof.numShapesPerElem())
  {
    throw std::runtime_error(
        "NavierStokesEquation currently requires equal-order fields");
  }
}

void NavierStokesEquation::checkParameters() const
{
  if (prm_.steps < 0 || prm_.dt <= 0.0
      || prm_.fluid.rho <= 0.0 || prm_.fluid.mu < 0.0
      || prm_.quad_order <= 0)
  {
    throw std::runtime_error(
        "NavierStokesEquation received invalid parameters");
  }
}

} // namespace femx
