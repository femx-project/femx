#include <stdexcept>
#include <utility>

#include <femx/assembly/Assembler.hpp>
#include <femx/assembly/FEMResidual.hpp>
#include <femx/problem/Linearization.hpp>

using namespace std;
using namespace femx::problem;
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
  checkCellCounts();
}

FEMResidual::FEMResidual(DofLayout            state_layout,
                         DofLayout            param_layout,
                         const ElementKernel& ker)
  : FEMResidual(state_layout, state_layout, param_layout, ker)
{
}

Dimensions FEMResidual::dims() const
{
  return {state_layout_.numDofs(), param_layout_.numDofs(), res_layout_.numDofs()};
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
  for (Index ic = 0; ic < numCells(); ++ic)
  {
    gather(state_layout_, state, ic, state_e);
    gather(param_layout_, prm, ic, prm_e);
    kernel_.res(ic, state_e, prm_e, res_e);
    assembler.addVec(ic, res_e, out);
  }
}

void FEMResidual::linearize(const Vector<Real>& state,
                            const Vector<Real>& prm,
                            Linearization&      out) const
{
  auto* matrix_out = dynamic_cast<MatrixLinearization*>(&out);
  if (matrix_out == nullptr)
  {
    throw runtime_error(
        "FEMResidual requires MatrixLinearization output");
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
                                   MatrixBuilder&      out) const
{
  checkGlobalSizes(state, prm);

  Assembler assembler(res_layout_, state_layout_);
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

void FEMResidual::assembleParamJac(const Vector<Real>& state,
                                   const Vector<Real>& prm,
                                   MatrixBuilder&      out) const
{
  checkGlobalSizes(state, prm);

  Assembler assembler(res_layout_, param_layout_);
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

void FEMResidual::checkCellCounts() const
{
  if (res_layout_.numElems() != state_layout_.numElems()
      || res_layout_.numElems() != param_layout_.numElems())
  {
    throw runtime_error(
        "FEMResidual layouts have different cell counts");
  }
}

void FEMResidual::checkGlobalSizes(const Vector<Real>& state,
                                   const Vector<Real>& prm) const
{
  const Dimensions dm = dims();
  if (state.size() != dm.nst)
  {
    throw runtime_error("FEMResidual state size mismatch");
  }
  if (prm.size() != dm.nprm)
  {
    throw runtime_error("FEMResidual parameter size mismatch");
  }
}

void FEMResidual::gather(const DofLayout&    lyt,
                         const Vector<Real>& global,
                         Index               ic,
                         Vector<Real>&       local)
{
  Vector<Index> dofs;
  lyt.elemDofs(ic, dofs);
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
      throw runtime_error("FEMResidual dof is out of range");
    }
    local[i] = global[dof];
  }
}

BoundaryFEMResidual::BoundaryFEMResidual(
    const Residual&              volume,
    BoundaryDofLayout            res_layout,
    BoundaryDofLayout            state_layout,
    BoundaryDofLayout            param_layout,
    const BoundaryElementKernel& ker)
  : volume_(volume),
    res_layout_(std::move(res_layout)),
    state_layout_(std::move(state_layout)),
    param_layout_(std::move(param_layout)),
    kernel_(ker)
{
  checkDimensions();
  checkFacetCompatibility();
}

Dimensions BoundaryFEMResidual::dims() const
{
  return volume_.dims();
}

void BoundaryFEMResidual::res(const Vector<Real>& state,
                              const Vector<Real>& prm,
                              Vector<Real>&       out) const
{
  volume_.res(state, prm, out);
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

void BoundaryFEMResidual::linearize(const Vector<Real>& state,
                                    const Vector<Real>& prm,
                                    Linearization&      out) const
{
  volume_.linearize(state, prm, out);

  auto* matrix_out = dynamic_cast<MatrixLinearization*>(&out);
  if (matrix_out == nullptr)
  {
    throw runtime_error(
        "BoundaryFEMResidual requires MatrixLinearization output");
  }

  addStateJac(state, prm, matrix_out->stateMatrix());
  matrix_out->stateMatrix().finalize();

  addParamJac(state, prm, matrix_out->paramMatrix());
  matrix_out->paramMatrix().finalize();
}

void BoundaryFEMResidual::addStateJac(const Vector<Real>& state,
                                      const Vector<Real>& prm,
                                      MatrixBuilder&      out) const
{
  const Dimensions dm = dims();
  if (out.numRows() != dm.nres || out.numCols() != dm.nst)
  {
    throw runtime_error(
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
    addMat(res_layout_, state_layout_, ib, jac_b, out);
  }
}

void BoundaryFEMResidual::addParamJac(const Vector<Real>& state,
                                      const Vector<Real>& prm,
                                      MatrixBuilder&      out) const
{
  const Dimensions dm = dims();
  if (out.numRows() != dm.nres || out.numCols() != dm.nprm)
  {
    throw runtime_error(
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
    addMat(res_layout_, param_layout_, ib, jac_b, out);
  }
}

void BoundaryFEMResidual::checkDimensions() const
{
  const Dimensions dm = dims();
  if (res_layout_.numDofs() != dm.nres
      || state_layout_.numDofs() != dm.nst
      || param_layout_.numDofs() != dm.nprm)
  {
    throw runtime_error(
        "BoundaryFEMResidual received inconsistent dimensions");
  }
}

void BoundaryFEMResidual::checkFacetCompatibility() const
{
  if (state_layout_.numFacets() != res_layout_.numFacets()
      || param_layout_.numFacets() != res_layout_.numFacets())
  {
    throw runtime_error(
        "BoundaryFEMResidual layouts have different facet counts");
  }

  for (Index ib = 0; ib < res_layout_.numFacets(); ++ib)
  {
    if (state_layout_.meshFacetIndex(ib) != res_layout_.meshFacetIndex(ib)
        || param_layout_.meshFacetIndex(ib)
               != res_layout_.meshFacetIndex(ib))
    {
      throw runtime_error(
          "BoundaryFEMResidual layouts select different facets");
    }
  }
}

void BoundaryFEMResidual::checkGlobalSizes(const Vector<Real>& state,
                                           const Vector<Real>& prm,
                                           const Vector<Real>& res_out) const
{
  const Dimensions dm = dims();
  if (state.size() != dm.nst || prm.size() != dm.nprm
      || res_out.size() != dm.nres)
  {
    throw runtime_error("BoundaryFEMResidual size mismatch");
  }
}

void BoundaryFEMResidual::gather(const BoundaryDofLayout& lyt,
                                 const Vector<Real>&      global,
                                 Index                    ib,
                                 Vector<Real>&            local)
{
  Vector<Index> dofs;
  lyt.facetDofs(ib, dofs);
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

void BoundaryFEMResidual::addVec(const BoundaryDofLayout& lyt,
                                 Index                    ib,
                                 const Vector<Real>&      local,
                                 Vector<Real>&            out)
{
  Vector<Index> dofs;
  lyt.facetDofs(ib, dofs);
  if (local.size() != dofs.size())
  {
    throw runtime_error(
        "BoundaryFEMResidual local residual size mismatch");
  }

  for (Index i = 0; i < local.size(); ++i)
  {
    const Index row = dofs[i];
    checkDof(row, out.size());
    out[row] += local[i];
  }
}

void BoundaryFEMResidual::addMat(const BoundaryDofLayout& row_layout,
                                 const BoundaryDofLayout& col_layout,
                                 Index                    ib,
                                 const DenseMatrix&       local,
                                 MatrixBuilder&           out)
{
  Vector<Index> row_dofs;
  Vector<Index> col_dofs;
  row_layout.facetDofs(ib, row_dofs);
  col_layout.facetDofs(ib, col_dofs);

  if (local.rows() != row_dofs.size() || local.cols() != col_dofs.size())
  {
    throw runtime_error(
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
    throw runtime_error("BoundaryFEMResidual dof is out of range");
  }
}

} // namespace assembly
} // namespace femx
