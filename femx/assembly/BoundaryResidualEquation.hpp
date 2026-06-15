#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

#include <femx/assembly/BoundaryDofLayout.hpp>
#include <femx/assembly/BoundaryElementKernel.hpp>
#include <femx/common/Types.hpp>
#include <femx/eq/AssembledResidualEquation.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/SystemMatrix.hpp>

namespace femx
{
namespace assembly
{

/** @brief Adds boundary-facet residual terms to an assembled equation. */
class BoundaryResidualEquation final : public eq::AssembledResidualEquation
{
public:
  BoundaryResidualEquation(const eq::AssembledResidualEquation& interior,
                           BoundaryDofLayout                   res_layout,
                           BoundaryDofLayout                   state_layout,
                           BoundaryDofLayout                   param_layout,
                           const BoundaryElementKernel&         kernel)
    : interior_(interior),
      res_layout_(std::move(res_layout)),
      state_layout_(std::move(state_layout)),
      param_layout_(std::move(param_layout)),
      kernel_(kernel)
  {
    checkDimensions();
    checkFacetCompatibility();
  }

  Index numStates() const override
  {
    return interior_.numStates();
  }

  Index numParams() const override
  {
    return interior_.numParams();
  }

  Index numRes() const override
  {
    return interior_.numRes();
  }

  void res(const Vector& state,
           const Vector& params,
           Vector&       out) const override
  {
    interior_.res(state, params, out);
    checkGlobalSizes(state, params, out);

    Vector state_b;
    Vector params_b;
    Vector res_b;
    for (Index ib = 0; ib < res_layout_.numFacets(); ++ib)
    {
      gather(state_layout_, state, ib, state_b);
      gather(param_layout_, params, ib, params_b);
      kernel_.res(ib, res_layout_.facet(ib), state_b, params_b, res_b);
      addVec(res_layout_, ib, res_b, out);
    }
  }

  void assembleStateJac(const Vector&         state,
                        const Vector&         params,
                        system::SystemMatrix& out) const override
  {
    interior_.assembleStateJac(state, params, out);
    if (out.numRows() != numRes() || out.numCols() != numStates())
    {
      throw std::runtime_error(
          "BoundaryResidualEquation state Jacobian size mismatch");
    }

    Vector      state_b;
    Vector      params_b;
    DenseMatrix jac_b;
    for (Index ib = 0; ib < res_layout_.numFacets(); ++ib)
    {
      gather(state_layout_, state, ib, state_b);
      gather(param_layout_, params, ib, params_b);
      kernel_.stateJac(
          ib, res_layout_.facet(ib), state_b, params_b, jac_b);
      addMat(res_layout_, state_layout_, ib, jac_b, out);
    }
  }

  void assembleParamJac(const Vector&         state,
                        const Vector&         params,
                        system::SystemMatrix& out) const override
  {
    interior_.assembleParamJac(state, params, out);
    if (out.numRows() != numRes() || out.numCols() != numParams())
    {
      throw std::runtime_error(
          "BoundaryResidualEquation parameter Jacobian size mismatch");
    }

    Vector      state_b;
    Vector      params_b;
    DenseMatrix jac_b;
    for (Index ib = 0; ib < res_layout_.numFacets(); ++ib)
    {
      gather(state_layout_, state, ib, state_b);
      gather(param_layout_, params, ib, params_b);
      kernel_.paramJac(
          ib, res_layout_.facet(ib), state_b, params_b, jac_b);
      addMat(res_layout_, param_layout_, ib, jac_b, out);
    }
  }

private:
  void checkDimensions() const
  {
    if (res_layout_.numDofs() != numRes()
        || state_layout_.numDofs() != numStates()
        || param_layout_.numDofs() != numParams())
    {
      throw std::runtime_error(
          "BoundaryResidualEquation received inconsistent dimensions");
    }
  }

  void checkFacetCompatibility() const
  {
    if (state_layout_.numFacets() != res_layout_.numFacets()
        || param_layout_.numFacets() != res_layout_.numFacets())
    {
      throw std::runtime_error(
          "BoundaryResidualEquation layouts have different facet counts");
    }

    for (Index ib = 0; ib < res_layout_.numFacets(); ++ib)
    {
      if (state_layout_.meshFacetIndex(ib) != res_layout_.meshFacetIndex(ib)
          || param_layout_.meshFacetIndex(ib)
                 != res_layout_.meshFacetIndex(ib))
      {
        throw std::runtime_error(
            "BoundaryResidualEquation layouts select different facets");
      }
    }
  }

  void checkGlobalSizes(const Vector& state,
                        const Vector& params,
                        const Vector& res_out) const
  {
    if (state.size() != numStates() || params.size() != numParams()
        || res_out.size() != numRes())
    {
      throw std::runtime_error("BoundaryResidualEquation size mismatch");
    }
  }

  static void gather(const BoundaryDofLayout& layout,
                     const Vector&            global,
                     Index                    ib,
                     Vector&                  local)
  {
    std::vector<Index> dofs;
    layout.facetDofs(ib, dofs);
    if (local.size() != static_cast<Index>(dofs.size()))
    {
      local.resize(static_cast<Index>(dofs.size()));
    }
    else
    {
      local.setZero();
    }

    for (Index i = 0; i < local.size(); ++i)
    {
      const Index dof = dofs[static_cast<std::size_t>(i)];
      checkDof(dof, global.size());
      local[i] = global[dof];
    }
  }

  static void addVec(const BoundaryDofLayout& layout,
                     Index                    ib,
                     const Vector&            local,
                     Vector&                  out)
  {
    std::vector<Index> dofs;
    layout.facetDofs(ib, dofs);
    if (local.size() != static_cast<Index>(dofs.size()))
    {
      throw std::runtime_error(
          "BoundaryResidualEquation local residual size mismatch");
    }

    for (Index i = 0; i < local.size(); ++i)
    {
      const Index row = dofs[static_cast<std::size_t>(i)];
      checkDof(row, out.size());
      out[row] += local[i];
    }
  }

  static void addMat(const BoundaryDofLayout& row_layout,
                     const BoundaryDofLayout& col_layout,
                     Index                    ib,
                     const DenseMatrix&       local,
                     system::SystemMatrix&    out)
  {
    std::vector<Index> row_dofs;
    std::vector<Index> col_dofs;
    row_layout.facetDofs(ib, row_dofs);
    col_layout.facetDofs(ib, col_dofs);

    if (local.rows() != static_cast<Index>(row_dofs.size())
        || local.cols() != static_cast<Index>(col_dofs.size()))
    {
      throw std::runtime_error(
          "BoundaryResidualEquation local matrix size mismatch");
    }

    for (Index i = 0; i < local.rows(); ++i)
    {
      const Index row = row_dofs[static_cast<std::size_t>(i)];
      checkDof(row, out.numRows());
      for (Index j = 0; j < local.cols(); ++j)
      {
        const Index col = col_dofs[static_cast<std::size_t>(j)];
        checkDof(col, out.numCols());
        out.add(row, col, local(i, j));
      }
    }
  }

  static void checkDof(Index dof, Index size)
  {
    if (dof < 0 || dof >= size)
    {
      throw std::runtime_error(
          "BoundaryResidualEquation dof is out of range");
    }
  }

private:
  const eq::AssembledResidualEquation& interior_;
  BoundaryDofLayout                    res_layout_;
  BoundaryDofLayout                    state_layout_;
  BoundaryDofLayout                    param_layout_;
  const BoundaryElementKernel&         kernel_;
};

} // namespace assembly
} // namespace femx
