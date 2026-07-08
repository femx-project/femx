#include <stdexcept>

#include <femx/assembly/Assembler.hpp>
#include <femx/assembly/FEMResidual.hpp>
#include <femx/state/Linearization.hpp>

using namespace std;
using namespace femx::state;
using namespace femx::linalg;

namespace femx
{
namespace assembly
{

FEMResidual::FEMResidual(DofLayout            res_layout,
                         DofLayout            state_layout,
                         DofLayout            param_layout,
                         const ElementKernel& ker)
  : res_layout_(res_layout),
    state_layout_(state_layout),
    param_layout_(param_layout),
    kernel_(ker)
{
  checkElemCounts();
}

FEMResidual::FEMResidual(DofLayout            state_layout,
                         DofLayout            param_layout,
                         const ElementKernel& ker)
  : FEMResidual(state_layout, state_layout, param_layout, ker)
{
}

FEMResidual::FEMResidual(DofLayout            state_layout,
                         const ElementKernel& ker)
  : res_layout_(state_layout),
    state_layout_(state_layout),
    kernel_(ker)
{
  checkElemCounts();
}

Dimensions FEMResidual::dims() const
{
  const Index num_params =
      param_layout_ ? param_layout_->numDofs() : 0;
  return {state_layout_.numDofs(), num_params, res_layout_.numDofs()};
}

void FEMResidual::res(const Vector<Real>& state,
                      const Vector<Real>& prm,
                      Vector<Real>&       out) const
{
  checkGlobalSizes(state, prm);

  Assembler assembler(res_layout_);
  assembler.initVec(out);

  Vector<Real> state_e;
  Vector<Real> prm_e;
  Vector<Real> res_e;
  for (Index ie = 0; ie < numElems(); ++ie)
  {
    gather(state_layout_, state, ie, state_e);
    if (param_layout_)
    {
      gather(*param_layout_, prm, ie, prm_e);
    }
    else
    {
      prm_e.clear();
    }
    kernel_.res(ie, state_e, prm_e, res_e);
    assembler.addVec(ie, res_e, out);
  }
}

void FEMResidual::linearize(const Vector<Real>& state,
                            const Vector<Real>& prm,
                            Linearization&      out) const
{
  auto* mat_out = dynamic_cast<MatrixLinearization*>(&out);
  if (mat_out == nullptr)
  {
    throw runtime_error(
        "FEMResidual requires MatrixLinearization output");
  }

  assembleStateJac(state, prm, mat_out->stateMat());
  mat_out->stateMat().finalize();

  assembleParamJac(state, prm, mat_out->paramMat());
  mat_out->paramMat().finalize();
}

Index FEMResidual::numElems() const
{
  return res_layout_.numElems();
}

void FEMResidual::assembleStateJac(const Vector<Real>& state,
                                   const Vector<Real>& prm,
                                   MatrixBuilder&      out) const
{
  checkGlobalSizes(state, prm);

  Assembler assembler(res_layout_, state_layout_);
  assembler.initMat(out);

  Vector<Real> state_e;
  Vector<Real> prm_e;
  DenseMatrix  J_e;
  for (Index ie = 0; ie < numElems(); ++ie)
  {
    gather(state_layout_, state, ie, state_e);
    if (param_layout_)
    {
      gather(*param_layout_, prm, ie, prm_e);
    }
    else
    {
      prm_e.clear();
    }
    kernel_.stateJac(ie, state_e, prm_e, J_e);
    assembler.addMat(ie, J_e, out);
  }
}

void FEMResidual::assembleParamJac(const Vector<Real>& state,
                                   const Vector<Real>& prm,
                                   MatrixBuilder&      out) const
{
  checkGlobalSizes(state, prm);

  if (!param_layout_)
  {
    out.resize(res_layout_.numDofs(), 0);
    out.setZero();
    return;
  }

  Assembler assembler(res_layout_, *param_layout_);
  assembler.initMat(out);

  Vector<Real> state_e;
  Vector<Real> prm_e;
  DenseMatrix  J_e;
  for (Index ie = 0; ie < numElems(); ++ie)
  {
    gather(state_layout_, state, ie, state_e);
    gather(*param_layout_, prm, ie, prm_e);
    kernel_.paramJac(ie, state_e, prm_e, J_e);
    assembler.addMat(ie, J_e, out);
  }
}

void FEMResidual::checkElemCounts() const
{
  if (res_layout_.numElems() != state_layout_.numElems()
      || (param_layout_ && res_layout_.numElems() != param_layout_->numElems()))
  {
    throw runtime_error(
        "FEMResidual layouts have different elem counts");
  }
}

void FEMResidual::checkGlobalSizes(const Vector<Real>& state,
                                   const Vector<Real>& prm) const
{
  const Dimensions dm = dims();
  if (state.size() != dm.num_states)
  {
    throw runtime_error("FEMResidual state size mismatch");
  }
  if (prm.size() != dm.num_params)
  {
    throw runtime_error("FEMResidual parameter size mismatch");
  }
}

void FEMResidual::gather(const DofLayout&    lyt,
                         const Vector<Real>& global,
                         Index               ie,
                         Vector<Real>&       local)
{
  Vector<Index> dofs;
  lyt.elemDofs(ie, dofs);
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
    const Index id = dofs[i];
    if (id < 0 || id >= global.size())
    {
      throw runtime_error("FEMResidual id is out of range");
    }
    local[i] = global[id];
  }
}

} // namespace assembly
} // namespace femx
