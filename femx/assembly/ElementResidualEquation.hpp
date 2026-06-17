#pragma once

#include <stdexcept>
#include <vector>

#include <femx/assembly/DofLayout.hpp>
#include <femx/assembly/ElementKernel.hpp>
#include <femx/assembly/SystemAssembler.hpp>
#include <femx/common/Types.hpp>
#include <femx/eq/AssembledResidualEquation.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/SystemMatrix.hpp>

namespace femx
{
namespace assembly
{

/** @brief AssembledResidualEquation assembled from cell-local FEM kernels. */
class ElementResidualEquation final : public eq::AssembledResidualEquation
{
public:
  ElementResidualEquation(DofLayout            res_layout,
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

  ElementResidualEquation(DofLayout            state_layout,
                          DofLayout            param_layout,
                          const ElementKernel& kernel)
    : ElementResidualEquation(state_layout, state_layout, param_layout, kernel)
  {
  }

  Index numStates() const override
  {
    return state_layout_.numDofs();
  }

  Index numParams() const override
  {
    return param_layout_.numDofs();
  }

  Index numRes() const override
  {
    return res_layout_.numDofs();
  }

  void res(const Vector<Real>& state,
           const Vector<Real>& params,
           Vector<Real>&       out) const override
  {
    checkGlobalSizes(state, params);

    SystemAssembler assembler(res_layout_);
    assembler.initVec(out);

    Vector<Real> state_e;
    Vector<Real> params_e;
    Vector<Real> res_e;
    for (Index ic = 0; ic < numCells(); ++ic)
    {
      gather(state_layout_, state, ic, state_e);
      gather(param_layout_, params, ic, params_e);
      kernel_.res(ic, state_e, params_e, res_e);
      assembler.addVec(ic, res_e, out);
    }
  }

  void assembleStateJac(const Vector<Real>&   state,
                        const Vector<Real>&   params,
                        system::SystemMatrix& out) const override
  {
    checkGlobalSizes(state, params);

    SystemAssembler assembler(res_layout_, state_layout_);
    assembler.initMat(out);

    Vector<Real> state_e;
    Vector<Real> params_e;
    DenseMatrix  jac_e;
    for (Index ic = 0; ic < numCells(); ++ic)
    {
      gather(state_layout_, state, ic, state_e);
      gather(param_layout_, params, ic, params_e);
      kernel_.stateJac(ic, state_e, params_e, jac_e);
      assembler.addMat(ic, jac_e, out);
    }
  }

  void assembleParamJac(const Vector<Real>&   state,
                        const Vector<Real>&   params,
                        system::SystemMatrix& out) const override
  {
    checkGlobalSizes(state, params);

    SystemAssembler assembler(res_layout_, param_layout_);
    assembler.initMat(out);

    Vector<Real> state_e;
    Vector<Real> params_e;
    DenseMatrix  jac_e;
    for (Index ic = 0; ic < numCells(); ++ic)
    {
      gather(state_layout_, state, ic, state_e);
      gather(param_layout_, params, ic, params_e);
      kernel_.paramJac(ic, state_e, params_e, jac_e);
      assembler.addMat(ic, jac_e, out);
    }
  }

private:
  Index numCells() const
  {
    return res_layout_.numElems();
  }

  void checkCellCounts() const
  {
    if (res_layout_.numElems() != state_layout_.numElems()
        || res_layout_.numElems() != param_layout_.numElems())
    {
      throw std::runtime_error(
          "ElementResidualEquation layouts have different cell counts");
    }
  }

  void checkGlobalSizes(const Vector<Real>& state, const Vector<Real>& params) const
  {
    if (state.size() != numStates())
    {
      throw std::runtime_error("ElementResidualEquation state size mismatch");
    }
    if (params.size() != numParams())
    {
      throw std::runtime_error("ElementResidualEquation parameter size mismatch");
    }
  }

  static void gather(const DofLayout&    layout,
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
        throw std::runtime_error(
            "ElementResidualEquation dof is out of range");
      }
      local[i] = global[dof];
    }
  }

private:
  DofLayout            res_layout_;
  DofLayout            state_layout_;
  DofLayout            param_layout_;
  const ElementKernel& kernel_;
};

} // namespace assembly
} // namespace femx
