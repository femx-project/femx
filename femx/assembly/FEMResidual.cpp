#include <stdexcept>
#include <utility>

#include <femx/assembly/Assembler.hpp>
#include <femx/assembly/FEMResidual.hpp>
#include <femx/problem/Linearization.hpp>

namespace femx
{
namespace assembly
{

FEMResidual::FEMResidual(DofLayout res_layout,
                         DofLayout state_layout,
                         DofLayout param_layout,
                         const ElementKernel& kernel)
  : res_layout_(res_layout),
    state_layout_(state_layout),
    param_layout_(param_layout),
    kernel_(kernel)
{
  checkCellCounts();
}

FEMResidual::FEMResidual(DofLayout state_layout,
                         DofLayout param_layout,
                         const ElementKernel& kernel)
  : FEMResidual(state_layout, state_layout, param_layout, kernel)
{
}

problem::Dimensions FEMResidual::dimensions() const
{
  return {state_layout_.numDofs(), param_layout_.numDofs(),
          res_layout_.numDofs()};
}

void FEMResidual::residual(const Vector<Real>& state,
                           const Vector<Real>& prm,
                           Vector<Real>& out) const
{
  checkGlobalSizes(state, prm);

  Assembler assembler(res_layout_);
  assembler.initVector(out);

  Vector<Real> state_e;
  Vector<Real> prm_e;
  Vector<Real> res_e;
  for (Index ic = 0; ic < numCells(); ++ic)
  {
    gather(state_layout_, state, ic, state_e);
    gather(param_layout_, prm, ic, prm_e);
    kernel_.res(ic, state_e, prm_e, res_e);
    assembler.addVector(ic, res_e, out);
  }
}

void FEMResidual::linearize(const Vector<Real>& state,
                            const Vector<Real>& prm,
                            problem::Linearization& out) const
{
  auto* matrix_out = dynamic_cast<problem::MatrixLinearization*>(&out);
  if (matrix_out == nullptr)
  {
    throw std::runtime_error(
        "FEMResidual requires problem::MatrixLinearization output");
  }

  assembleStateJac(state, prm, matrix_out->stateMatrix());
  matrix_out->stateMatrix().finalize();

  assembleParamJac(state, prm, matrix_out->paramMatrix());
  matrix_out->paramMatrix().finalize();
}

Index FEMResidual::numCells() const
{
  return res_layout_.numElems();
}

void FEMResidual::assembleStateJac(const Vector<Real>& state,
                                   const Vector<Real>& prm,
                                   algebra::MatrixBuilder& out) const
{
  checkGlobalSizes(state, prm);

  Assembler assembler(res_layout_, state_layout_);
  assembler.initMatrix(out);

  Vector<Real> state_e;
  Vector<Real> prm_e;
  DenseMatrix  jac_e;
  for (Index ic = 0; ic < numCells(); ++ic)
  {
    gather(state_layout_, state, ic, state_e);
    gather(param_layout_, prm, ic, prm_e);
    kernel_.stateJac(ic, state_e, prm_e, jac_e);
    assembler.addMatrix(ic, jac_e, out);
  }
}

void FEMResidual::assembleParamJac(const Vector<Real>& state,
                                   const Vector<Real>& prm,
                                   algebra::MatrixBuilder& out) const
{
  checkGlobalSizes(state, prm);

  Assembler assembler(res_layout_, param_layout_);
  assembler.initMatrix(out);

  Vector<Real> state_e;
  Vector<Real> prm_e;
  DenseMatrix  jac_e;
  for (Index ic = 0; ic < numCells(); ++ic)
  {
    gather(state_layout_, state, ic, state_e);
    gather(param_layout_, prm, ic, prm_e);
    kernel_.paramJac(ic, state_e, prm_e, jac_e);
    assembler.addMatrix(ic, jac_e, out);
  }
}

void FEMResidual::checkCellCounts() const
{
  if (res_layout_.numElems() != state_layout_.numElems()
      || res_layout_.numElems() != param_layout_.numElems())
  {
    throw std::runtime_error(
        "FEMResidual layouts have different cell counts");
  }
}

void FEMResidual::checkGlobalSizes(const Vector<Real>& state,
                                   const Vector<Real>& prm) const
{
  const problem::Dimensions dims = dimensions();
  if (state.size() != dims.num_states)
  {
    throw std::runtime_error("FEMResidual state size mismatch");
  }
  if (prm.size() != dims.num_params)
  {
    throw std::runtime_error("FEMResidual parameter size mismatch");
  }
}

void FEMResidual::gather(const DofLayout& layout,
                         const Vector<Real>& global,
                         Index ic,
                         Vector<Real>& local)
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
      throw std::runtime_error("FEMResidual dof is out of range");
    }
    local[i] = global[dof];
  }
}

BoundaryFEMResidual::BoundaryFEMResidual(
    const problem::Residual& volume,
    BoundaryDofLayout res_layout,
    BoundaryDofLayout state_layout,
    BoundaryDofLayout param_layout,
    const BoundaryElementKernel& kernel)
  : volume_(volume),
    res_layout_(std::move(res_layout)),
    state_layout_(std::move(state_layout)),
    param_layout_(std::move(param_layout)),
    kernel_(kernel)
{
  checkDimensions();
  checkFacetCompatibility();
}

problem::Dimensions BoundaryFEMResidual::dimensions() const
{
  return volume_.dimensions();
}

void BoundaryFEMResidual::residual(const Vector<Real>& state,
                                   const Vector<Real>& prm,
                                   Vector<Real>& out) const
{
  volume_.residual(state, prm, out);
  checkGlobalSizes(state, prm, out);

  Vector<Real> state_b;
  Vector<Real> prm_b;
  Vector<Real> res_b;
  for (Index ib = 0; ib < res_layout_.numFacets(); ++ib)
  {
    gather(state_layout_, state, ib, state_b);
    gather(param_layout_, prm, ib, prm_b);
    kernel_.res(ib, res_layout_.facet(ib), state_b, prm_b, res_b);
    addVector(res_layout_, ib, res_b, out);
  }
}

void BoundaryFEMResidual::linearize(const Vector<Real>& state,
                                    const Vector<Real>& prm,
                                    problem::Linearization& out) const
{
  volume_.linearize(state, prm, out);

  auto* matrix_out = dynamic_cast<problem::MatrixLinearization*>(&out);
  if (matrix_out == nullptr)
  {
    throw std::runtime_error(
        "BoundaryFEMResidual requires problem::MatrixLinearization output");
  }

  addStateJac(state, prm, matrix_out->stateMatrix());
  matrix_out->stateMatrix().finalize();

  addParamJac(state, prm, matrix_out->paramMatrix());
  matrix_out->paramMatrix().finalize();
}

void BoundaryFEMResidual::addStateJac(const Vector<Real>& state,
                                      const Vector<Real>& prm,
                                      algebra::MatrixBuilder& out) const
{
  const problem::Dimensions dims = dimensions();
  if (out.numRows() != dims.num_residuals || out.numCols() != dims.num_states)
  {
    throw std::runtime_error(
        "BoundaryFEMResidual state Jacobian size mismatch");
  }

  Vector<Real> state_b;
  Vector<Real> prm_b;
  DenseMatrix  jac_b;
  for (Index ib = 0; ib < res_layout_.numFacets(); ++ib)
  {
    gather(state_layout_, state, ib, state_b);
    gather(param_layout_, prm, ib, prm_b);
    kernel_.stateJac(ib, res_layout_.facet(ib), state_b, prm_b, jac_b);
    addMatrix(res_layout_, state_layout_, ib, jac_b, out);
  }
}

void BoundaryFEMResidual::addParamJac(const Vector<Real>& state,
                                      const Vector<Real>& prm,
                                      algebra::MatrixBuilder& out) const
{
  const problem::Dimensions dims = dimensions();
  if (out.numRows() != dims.num_residuals || out.numCols() != dims.num_params)
  {
    throw std::runtime_error(
        "BoundaryFEMResidual parameter Jacobian size mismatch");
  }

  Vector<Real> state_b;
  Vector<Real> prm_b;
  DenseMatrix  jac_b;
  for (Index ib = 0; ib < res_layout_.numFacets(); ++ib)
  {
    gather(state_layout_, state, ib, state_b);
    gather(param_layout_, prm, ib, prm_b);
    kernel_.paramJac(ib, res_layout_.facet(ib), state_b, prm_b, jac_b);
    addMatrix(res_layout_, param_layout_, ib, jac_b, out);
  }
}

void BoundaryFEMResidual::checkDimensions() const
{
  const problem::Dimensions dims = dimensions();
  if (res_layout_.numDofs() != dims.num_residuals
      || state_layout_.numDofs() != dims.num_states
      || param_layout_.numDofs() != dims.num_params)
  {
    throw std::runtime_error(
        "BoundaryFEMResidual received inconsistent dimensions");
  }
}

void BoundaryFEMResidual::checkFacetCompatibility() const
{
  if (state_layout_.numFacets() != res_layout_.numFacets()
      || param_layout_.numFacets() != res_layout_.numFacets())
  {
    throw std::runtime_error(
        "BoundaryFEMResidual layouts have different facet counts");
  }

  for (Index ib = 0; ib < res_layout_.numFacets(); ++ib)
  {
    if (state_layout_.meshFacetIndex(ib) != res_layout_.meshFacetIndex(ib)
        || param_layout_.meshFacetIndex(ib)
               != res_layout_.meshFacetIndex(ib))
    {
      throw std::runtime_error(
          "BoundaryFEMResidual layouts select different facets");
    }
  }
}

void BoundaryFEMResidual::checkGlobalSizes(const Vector<Real>& state,
                                           const Vector<Real>& prm,
                                           const Vector<Real>& res_out) const
{
  const problem::Dimensions dims = dimensions();
  if (state.size() != dims.num_states || prm.size() != dims.num_params
      || res_out.size() != dims.num_residuals)
  {
    throw std::runtime_error("BoundaryFEMResidual size mismatch");
  }
}

void BoundaryFEMResidual::gather(const BoundaryDofLayout& layout,
                                 const Vector<Real>& global,
                                 Index ib,
                                 Vector<Real>& local)
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

void BoundaryFEMResidual::addVector(const BoundaryDofLayout& layout,
                                    Index ib,
                                    const Vector<Real>& local,
                                    Vector<Real>& out)
{
  Vector<Index> dofs;
  layout.facetDofs(ib, dofs);
  if (local.size() != dofs.size())
  {
    throw std::runtime_error(
        "BoundaryFEMResidual local residual size mismatch");
  }

  for (Index i = 0; i < local.size(); ++i)
  {
    const Index row = dofs[i];
    checkDof(row, out.size());
    out[row] += local[i];
  }
}

void BoundaryFEMResidual::addMatrix(const BoundaryDofLayout& row_layout,
                                    const BoundaryDofLayout& col_layout,
                                    Index ib,
                                    const DenseMatrix& local,
                                    algebra::MatrixBuilder& out)
{
  Vector<Index> row_dofs;
  Vector<Index> col_dofs;
  row_layout.facetDofs(ib, row_dofs);
  col_layout.facetDofs(ib, col_dofs);

  if (local.rows() != row_dofs.size() || local.cols() != col_dofs.size())
  {
    throw std::runtime_error(
        "BoundaryFEMResidual local matrix size mismatch");
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

void BoundaryFEMResidual::checkDof(Index dof, Index size)
{
  if (dof < 0 || dof >= size)
  {
    throw std::runtime_error("BoundaryFEMResidual dof is out of range");
  }
}

} // namespace assembly
} // namespace femx
