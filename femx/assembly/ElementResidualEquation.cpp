#include <stdexcept>

#include <femx/assembly/ElementResidualEquation.hpp>
#include <femx/assembly/SystemAssembler.hpp>

using namespace femx::system;

namespace femx
{
namespace assembly
{

ElementResidualEquation::ElementResidualEquation(
    DofLayout            res_layout,
    DofLayout            state_layout,
    DofLayout            param_layout,
    const ElementKernel& kernel)
  : res_layout_(res_layout),
    state_layout_(state_layout),
    param_layout_(param_layout),
    kernel_(kernel)
{
  checkCellCounts();
}

ElementResidualEquation::ElementResidualEquation(DofLayout            state_layout,
                                                 DofLayout            param_layout,
                                                 const ElementKernel& kernel)
  : ElementResidualEquation(state_layout, state_layout, param_layout, kernel)
{
}

Index ElementResidualEquation::numStates() const
{
  return state_layout_.numDofs();
}

Index ElementResidualEquation::numParams() const
{
  return param_layout_.numDofs();
}

Index ElementResidualEquation::numRes() const
{
  return res_layout_.numDofs();
}

void ElementResidualEquation::res(const Vector<Real>& state,
                                  const Vector<Real>& prm,
                                  Vector<Real>&       out) const
{
  checkGlobalSizes(state, prm);

  SystemAssembler assembler(res_layout_);
  assembler.initVec(out);

  Vector<Real> state_e;
  Vector<Real> prm_e;
  Vector<Real> res_e;
  for (Index ic = 0; ic < numCells(); ++ic)
  {
    gather(state_layout_, state, ic, state_e);
    gather(param_layout_, prm, ic, prm_e);
    kernel_.res(ic, state_e, prm_e, res_e);
    assembler.addVec(ic, res_e, out);
  }
}

void ElementResidualEquation::assembleStateJac(
    const Vector<Real>&   state,
    const Vector<Real>&   prm,
    SystemMatrix& out) const
{
  checkGlobalSizes(state, prm);

  SystemAssembler assembler(res_layout_, state_layout_);
  assembler.initMat(out);

  Vector<Real> state_e;
  Vector<Real> prm_e;
  DenseMatrix  jac_e;
  for (Index ic = 0; ic < numCells(); ++ic)
  {
    gather(state_layout_, state, ic, state_e);
    gather(param_layout_, prm, ic, prm_e);
    kernel_.stateJac(ic, state_e, prm_e, jac_e);
    assembler.addMat(ic, jac_e, out);
  }
}

void ElementResidualEquation::assembleParamJac(
    const Vector<Real>&   state,
    const Vector<Real>&   prm,
    SystemMatrix& out) const
{
  checkGlobalSizes(state, prm);

  SystemAssembler assembler(res_layout_, param_layout_);
  assembler.initMat(out);

  Vector<Real> state_e;
  Vector<Real> prm_e;
  DenseMatrix  jac_e;
  for (Index ic = 0; ic < numCells(); ++ic)
  {
    gather(state_layout_, state, ic, state_e);
    gather(param_layout_, prm, ic, prm_e);
    kernel_.paramJac(ic, state_e, prm_e, jac_e);
    assembler.addMat(ic, jac_e, out);
  }
}

Index ElementResidualEquation::numCells() const
{
  return res_layout_.numElems();
}

void ElementResidualEquation::checkCellCounts() const
{
  if (res_layout_.numElems() != state_layout_.numElems()
      || res_layout_.numElems() != param_layout_.numElems())
  {
    throw std::runtime_error(
        "ElementResidualEquation layouts have different cell counts");
  }
}

void ElementResidualEquation::checkGlobalSizes(
    const Vector<Real>& state,
    const Vector<Real>& prm) const
{
  if (state.size() != numStates())
  {
    throw std::runtime_error("ElementResidualEquation state size mismatch");
  }
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "ElementResidualEquation parameter size mismatch");
  }
}

void ElementResidualEquation::gather(const DofLayout&    layout,
                                     const Vector<Real>& global,
                                     Index               ic,
                                     Vector<Real>&       local)
{
  Vector<Index> dofs;
  layout.elemDofs(ic, dofs);
  if (local.size() != dofs.size())
  {
    local.resize(dofs.size());
  }
  else
  {
    local.setZero();
  }

  for (Index i = 0; i < local.size(); ++i)
  {
    const Index dof = dofs[i];
    if (dof < 0 || dof >= global.size())
    {
      throw std::runtime_error("ElementResidualEquation dof is out of range");
    }
    local[i] = global[dof];
  }
}

} // namespace assembly
} // namespace femx
