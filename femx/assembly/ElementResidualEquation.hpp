#pragma once

#include <stdexcept>
#include <vector>

#include <femx/core/Types.hpp>
#include <femx/system/SystemMatrix.hpp>
#include <femx/equation/AssembledResidualEquation.hpp>
#include <femx/assembly/DofLayout.hpp>
#include <femx/assembly/ElementKernel.hpp>
#include <femx/assembly/SystemAssembler.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace assembly
{

/** @brief AssembledResidualEquation assembled from cell-local FEM kernels. */
class ElementResidualEquation final : public equation::AssembledResidualEquation
{
public:
  ElementResidualEquation(DofLayout            residual_layout,
                          DofLayout            state_layout,
                          DofLayout            param_layout,
                          const ElementKernel& kernel)
    : residual_layout_(residual_layout),
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

  index_type numStates() const override
  {
    return state_layout_.numDofs();
  }

  index_type numParams() const override
  {
    return param_layout_.numDofs();
  }

  index_type numResiduals() const override
  {
    return residual_layout_.numDofs();
  }

  void residual(const Vector& state,
                const Vector& params,
                Vector&       out) const override
  {
    checkGlobalSizes(state, params);

    SystemAssembler assembler(residual_layout_);
    assembler.initVec(out);

    Vector state_e;
    Vector params_e;
    Vector residual_e;
    for (index_type cell = 0; cell < numCells(); ++cell)
    {
      gather(state_layout_, state, cell, state_e);
      gather(param_layout_, params, cell, params_e);
      kernel_.res(cell, state_e, params_e, residual_e);
      assembler.addVec(cell, residual_e, out);
    }
  }

  void assembleStateJac(const Vector& state,
                        const Vector& params,
                        system::SystemMatrix& out) const override
  {
    checkGlobalSizes(state, params);

    SystemAssembler assembler(residual_layout_, state_layout_);
    assembler.initMat(out);

    Vector      state_e;
    Vector      params_e;
    DenseMatrix jac_e;
    for (index_type cell = 0; cell < numCells(); ++cell)
    {
      gather(state_layout_, state, cell, state_e);
      gather(param_layout_, params, cell, params_e);
      kernel_.stateJac(cell, state_e, params_e, jac_e);
      assembler.addMat(cell, jac_e, out);
    }
  }

  void assembleParamJac(const Vector& state,
                        const Vector& params,
                        system::SystemMatrix& out) const override
  {
    checkGlobalSizes(state, params);

    SystemAssembler assembler(residual_layout_, param_layout_);
    assembler.initMat(out);

    Vector      state_e;
    Vector      params_e;
    DenseMatrix jac_e;
    for (index_type cell = 0; cell < numCells(); ++cell)
    {
      gather(state_layout_, state, cell, state_e);
      gather(param_layout_, params, cell, params_e);
      kernel_.paramJac(cell, state_e, params_e, jac_e);
      assembler.addMat(cell, jac_e, out);
    }
  }

private:
  index_type numCells() const
  {
    return residual_layout_.numElems();
  }

  void checkCellCounts() const
  {
    if (residual_layout_.numElems() != state_layout_.numElems()
        || residual_layout_.numElems() != param_layout_.numElems())
    {
      throw std::runtime_error(
          "ElementResidualEquation layouts have different cell counts");
    }
  }

  void checkGlobalSizes(const Vector& state, const Vector& params) const
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

  static void gather(const DofLayout& layout,
                     const Vector&    global,
                     index_type       cell,
                     Vector&          local)
  {
    std::vector<index_type> dofs;
    layout.elemDofs(cell, dofs);
    if (local.size() != static_cast<index_type>(dofs.size()))
    {
      local.resize(static_cast<index_type>(dofs.size()));
    }
    else
    {
      local.setZero();
    }

    for (index_type i = 0; i < local.size(); ++i)
    {
      const index_type dof = dofs[static_cast<std::size_t>(i)];
      if (dof < 0 || dof >= global.size())
      {
        throw std::runtime_error(
            "ElementResidualEquation dof is out of range");
      }
      local[i] = global[dof];
    }
  }

private:
  DofLayout            residual_layout_;
  DofLayout            state_layout_;
  DofLayout            param_layout_;
  const ElementKernel& kernel_;
};

} // namespace assembly
} // namespace femx
