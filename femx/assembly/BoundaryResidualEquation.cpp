#include <stdexcept>
#include <utility>

#include <femx/assembly/BoundaryResidualEquation.hpp>

using namespace femx::eq;
using namespace femx::system;

namespace femx
{
namespace assembly
{

BoundaryResidualEquation::BoundaryResidualEquation(
    const MatrixResidualEquation& volume_eq,
    BoundaryDofLayout             res_layout,
    BoundaryDofLayout             state_layout,
    BoundaryDofLayout             param_layout,
    const BoundaryElementKernel&  kernel)
  : volume_eq_(volume_eq),
    res_layout_(std::move(res_layout)),
    state_layout_(std::move(state_layout)),
    param_layout_(std::move(param_layout)),
    kernel_(kernel)
{
  checkDimensions();
  checkFacetCompatibility();
}

Index BoundaryResidualEquation::numStates() const
{
  return volume_eq_.numStates();
}

Index BoundaryResidualEquation::numParams() const
{
  return volume_eq_.numParams();
}

Index BoundaryResidualEquation::numRes() const
{
  return volume_eq_.numRes();
}

void BoundaryResidualEquation::res(const Vector<Real>& state,
                                   const Vector<Real>& prm,
                                   Vector<Real>&       out) const
{
  volume_eq_.res(state, prm, out);
  checkGlobalSizes(state, prm, out);

  Vector<Real> state_b;
  Vector<Real> prm_b;
  Vector<Real> res_b;
  for (Index ib = 0; ib < res_layout_.numFacets(); ++ib)
  {
    gather(state_layout_, state, ib, state_b);
    gather(param_layout_, prm, ib, prm_b);
    kernel_.res(ib, res_layout_.facet(ib), state_b, prm_b, res_b);
    addVec(res_layout_, ib, res_b, out);
  }
}

void BoundaryResidualEquation::assembleStateJac(
    const Vector<Real>& state,
    const Vector<Real>& prm,
    SystemMatrix&       out) const
{
  volume_eq_.assembleStateJac(state, prm, out);
  if (out.numRows() != numRes() || out.numCols() != numStates())
  {
    throw std::runtime_error(
        "BoundaryResidualEquation state Jacobian size mismatch");
  }

  Vector<Real> state_b;
  Vector<Real> prm_b;
  DenseMatrix  jac_b;
  for (Index ib = 0; ib < res_layout_.numFacets(); ++ib)
  {
    gather(state_layout_, state, ib, state_b);
    gather(param_layout_, prm, ib, prm_b);
    kernel_.stateJac(ib, res_layout_.facet(ib), state_b, prm_b, jac_b);
    addMat(res_layout_, state_layout_, ib, jac_b, out);
  }
}

void BoundaryResidualEquation::assembleParamJac(
    const Vector<Real>& state,
    const Vector<Real>& prm,
    SystemMatrix&       out) const
{
  volume_eq_.assembleParamJac(state, prm, out);
  if (out.numRows() != numRes() || out.numCols() != numParams())
  {
    throw std::runtime_error(
        "BoundaryResidualEquation parameter Jacobian size mismatch");
  }

  Vector<Real> state_b;
  Vector<Real> prm_b;
  DenseMatrix  jac_b;
  for (Index ib = 0; ib < res_layout_.numFacets(); ++ib)
  {
    gather(state_layout_, state, ib, state_b);
    gather(param_layout_, prm, ib, prm_b);
    kernel_.paramJac(ib, res_layout_.facet(ib), state_b, prm_b, jac_b);
    addMat(res_layout_, param_layout_, ib, jac_b, out);
  }
}

void BoundaryResidualEquation::checkDimensions() const
{
  if (res_layout_.numDofs() != numRes()
      || state_layout_.numDofs() != numStates()
      || param_layout_.numDofs() != numParams())
  {
    throw std::runtime_error(
        "BoundaryResidualEquation received inconsistent dimensions");
  }
}

void BoundaryResidualEquation::checkFacetCompatibility() const
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

void BoundaryResidualEquation::checkGlobalSizes(
    const Vector<Real>& state,
    const Vector<Real>& prm,
    const Vector<Real>& res_out) const
{
  if (state.size() != numStates() || prm.size() != numParams()
      || res_out.size() != numRes())
  {
    throw std::runtime_error("BoundaryResidualEquation size mismatch");
  }
}

void BoundaryResidualEquation::gather(const BoundaryDofLayout& layout,
                                      const Vector<Real>&      global,
                                      Index                    ib,
                                      Vector<Real>&            local)
{
  Vector<Index> dofs;
  layout.facetDofs(ib, dofs);
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
    checkDof(dof, global.size());
    local[i] = global[dof];
  }
}

void BoundaryResidualEquation::addVec(const BoundaryDofLayout& layout,
                                      Index                    ib,
                                      const Vector<Real>&      local,
                                      Vector<Real>&            out)
{
  Vector<Index> dofs;
  layout.facetDofs(ib, dofs);
  if (local.size() != dofs.size())
  {
    throw std::runtime_error(
        "BoundaryResidualEquation local residual size mismatch");
  }

  for (Index i = 0; i < local.size(); ++i)
  {
    const Index row = dofs[i];
    checkDof(row, out.size());
    out[row] += local[i];
  }
}

void BoundaryResidualEquation::addMat(const BoundaryDofLayout& row_layout,
                                      const BoundaryDofLayout& col_layout,
                                      Index                    ib,
                                      const DenseMatrix&       local,
                                      SystemMatrix&            out)
{
  Vector<Index> row_dofs;
  Vector<Index> col_dofs;
  row_layout.facetDofs(ib, row_dofs);
  col_layout.facetDofs(ib, col_dofs);

  if (local.rows() != row_dofs.size() || local.cols() != col_dofs.size())
  {
    throw std::runtime_error(
        "BoundaryResidualEquation local matrix size mismatch");
  }

  for (Index i = 0; i < local.rows(); ++i)
  {
    const Index row = row_dofs[i];
    checkDof(row, out.numRows());
    for (Index j = 0; j < local.cols(); ++j)
    {
      const Index col = col_dofs[j];
      checkDof(col, out.numCols());
      out.add(row, col, local(i, j));
    }
  }
}

void BoundaryResidualEquation::checkDof(Index dof,
                                        Index size)
{
  if (dof < 0 || dof >= size)
  {
    throw std::runtime_error("BoundaryResidualEquation dof is out of range");
  }
}

} // namespace assembly
} // namespace femx
