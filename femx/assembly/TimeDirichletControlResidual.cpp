#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/assembly/TimeDirichletControlResidual.hpp>
#include <femx/linalg/BlockVectorView.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/MatrixBuilder.hpp>
#include <femx/linalg/native/CsrAssemblyMatrix.hpp>

#if defined(FEMX_HAS_PETSC)
#include <femx/linalg/petsc/PETScAssemblyMatrix.hpp>
#endif
using namespace femx::state;
using namespace femx::linalg;

namespace femx
{
namespace assembly
{

namespace
{

void replaceSparseRow(CsrAssemblyMatrix& mat,
                      Index              row,
                      Real               diag)
{
  CsrMatrix& sparse = mat.mat();
  if (row < 0 || row >= sparse.rows())
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual sparse row is out of range");
  }

  const Index* rp   = sparse.rowPtrData();
  const Index* ci   = sparse.colIndData();
  Real*        vals = sparse.valuesData();

  bool has_diag = false;
  for (Index k = rp[row]; k < rp[row + 1]; ++k)
  {
    vals[k] = 0.0;
    if (ci[k] == row)
    {
      vals[k]  = diag;
      has_diag = true;
    }
  }
  if (diag != 0.0 && !has_diag)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual sparse pattern lacks diagonal");
  }
}

} // namespace

TimeDirichletControlResidual::TimeDirichletControlResidual(
    const TimeResidual&         base,
    fem::DirichletControl       ctr,
    Vector<Index>               fdofs,
    Index                       ctr_param_offset,
    Index                       num_params,
    Vector<Real>                fvals,
    Vector<LinearInterpolation> ctr_time_stencils)
  : base_(base),
    ctr_(std::move(ctr)),
    fdofs_(std::move(fdofs)),
    fvals_(std::move(fvals)),
    base_dims_(base.dims()),
    dims_(base_dims_),
    ctr_param_offset_(ctr_param_offset)
{
  if (base_dims_.num_residuals != base_dims_.num_states)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual requires square state residuals");
  }
  if (base_dims_.num_params != 0)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual requires a parameter-free base residual");
  }
  if (ctr_param_offset_ < 0)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual received negative parameter offset");
  }
  initializeControlTimeStencils(std::move(ctr_time_stencils));

  const Index required_ctr_params = ctr_param_levels_ * ctr_.numDofs();
  dims_.num_params =
      num_params < 0
          ? ctr_param_offset_ + required_ctr_params
          : num_params;
  if (dims_.num_params < ctr_param_offset_ + required_ctr_params)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual parameter count is too small");
  }
  base_prm_.resize(base_dims_.num_params);

  if (fvals_.empty())
  {
    fvals_.resize(fdofs_.size());
  }
  else if (fvals_.size() != fdofs_.size()
           && fvals_.size()
                  != base_dims_.num_steps * fdofs_.size())
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual fixed value size mismatch");
  }

  for (Index id : ctr_.stateDofs())
  {
    if (id < 0 || id >= base_dims_.num_states)
    {
      throw std::runtime_error(
          "TimeDirichletControlResidual control id is out of range");
    }
  }
  for (Index id : fdofs_)
  {
    if (id < 0 || id >= base_dims_.num_states)
    {
      throw std::runtime_error(
          "TimeDirichletControlResidual fixed id is out of range");
    }
    for (Index ctr_id : ctr_.stateDofs())
    {
      if (id == ctr_id)
      {
        throw std::runtime_error(
            "TimeDirichletControlResidual received overlapping dofs");
      }
    }
  }
}

TimeDims TimeDirichletControlResidual::dims() const
{
  return dims_;
}

Index TimeDirichletControlResidual::numSteps() const
{
  return dims_.num_steps;
}

Index TimeDirichletControlResidual::numStates() const
{
  return dims_.num_states;
}

Index TimeDirichletControlResidual::numParams() const
{
  return dims_.num_params;
}

Index TimeDirichletControlResidual::numRes() const
{
  return dims_.num_residuals;
}

void TimeDirichletControlResidual::res(
    const TimeContext& ctx,
    Vector<Real>&      out) const
{
  checkContext(ctx);

  TimeContext base_ctx = ctx;
  base_ctx.prm         = &base_prm_;
  base_.res(base_ctx, out);
  if (out.size() != dims_.num_residuals)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual base residual size mismatch");
  }

  for (Index i = 0; i < ctr_.numDofs(); ++i)
  {
    const Index row = ctr_.stateDof(i);
    out[row]        = (*ctx.nxt)[row] - ctrValue(ctx.step, i, *ctx.prm);
  }
  for (Index i = 0; i < fdofs_.size(); ++i)
  {
    const Index row = fdofs_[i];
    out[row]        = (*ctx.nxt)[row] - fixedValue(ctx.step, i);
  }
}

void TimeDirichletControlResidual::applyJac(
    const TimeContext&  ctx,
    VariableBlock       wrt,
    const Vector<Real>& dir,
    Vector<Real>&       out) const
{
  checkContext(ctx);
  if (wrt.isParam())
  {
    applyControlParamJac(ctx, dir, out);
    return;
  }

  TimeContext base_ctx = ctx;
  base_ctx.prm         = &base_prm_;
  base_.applyJac(base_ctx, wrt, dir, out);
  if (out.size() != dims_.num_residuals)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual Jacobian apply size mismatch");
  }

  const Real diag = wrt.isNextState() ? 1.0 : 0.0;
  for (Index row : constrainedRows())
  {
    out[row] = diag == 0.0 ? 0.0 : dir[row];
  }
}

void TimeDirichletControlResidual::applyJacT(
    const TimeContext&  ctx,
    VariableBlock       wrt,
    const Vector<Real>& adj,
    Vector<Real>&       out) const
{
  checkContext(ctx);
  if (wrt.isParam())
  {
    applyControlParamJacT(ctx, adj, out);
    return;
  }

  throw std::runtime_error(
      "TimeDirichletControlResidual state transpose apply requires assembled Jacobians");
}

bool TimeDirichletControlResidual::assembleJac(
    const TimeContext& ctx,
    VariableBlock      wrt,
    MatrixBuilder&     out) const
{
  checkContext(ctx);
  if (wrt.isParam())
  {
    out.resize(dims_.num_residuals, dims_.num_params);
    out.setZero();
    const LinearInterpolation& interp = ctr_time_stencils_[ctx.step];
    for (Index i = 0; i < ctr_.numDofs(); ++i)
    {
      const Index row = ctr_.stateDof(i);
      interp.forEachWeight(
          [&](Index level, Real wt)
          {
            out.set(row, ctrIndex(level, i), -wt);
          });
    }
    return true;
  }

  TimeContext base_ctx = ctx;
  base_ctx.prm         = &base_prm_;
  if (!base_.assembleJac(base_ctx, wrt, out))
  {
    return false;
  }

  replaceStateRows(out, wrt.isNextState() ? 1.0 : 0.0);
  return true;
}

void TimeDirichletControlResidual::prepareLinearSolve(
    const TimeContext& ctx,
    VariableBlock      wrt,
    MatrixBuilder&     J,
    Vector<Real>&      rhs) const
{
  checkContext(ctx);
  if (!wrt.isNextState())
  {
    return;
  }
  eliminateStateColumns(J, rhs);
}

const fem::DirichletControl& TimeDirichletControlResidual::control() const
{
  return ctr_;
}

void TimeDirichletControlResidual::checkContext(
    const TimeContext& ctx) const
{
  if (ctx.step < 0 || ctx.step >= dims_.num_steps)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual step is out of range");
  }
  const TimeHistoryView hist = ctx.hist;
  if (hist.count() < dims_.num_history_states
      || hist.stateSize() != dims_.num_states)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual history state size mismatch");
  }
  checkVector(ctx.nxt, dims_.num_states);
  checkVector(ctx.prm, dims_.num_params);
}

void TimeDirichletControlResidual::checkVector(const Vector<Real>* value,
                                               Index               size) const
{
  if (value == nullptr || value->size() != size)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual vector size mismatch");
  }
}

void TimeDirichletControlResidual::replaceStateRows(
    MatrixBuilder& out,
    Real           diag) const
{
  if (out.numRows() != dims_.num_residuals
      || out.numCols() != dims_.num_states)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual matrix size mismatch");
  }

  const Vector<Index> rows = constrainedRows();
  if (auto* sparse = dynamic_cast<CsrAssemblyMatrix*>(&out))
  {
    for (Index row : rows)
    {
      replaceSparseRow(*sparse, row, diag);
    }
    return;
  }

#if defined(FEMX_HAS_PETSC)
  if (auto* petsc = dynamic_cast<PETScAssemblyMatrix*>(&out))
  {
    out.finalize();
    petsc->zeroRows(rows, diag);
    return;
  }
#endif

  for (Index row : rows)
  {
    for (Index col = 0; col < out.numCols(); ++col)
    {
      out.set(row, col, 0.0);
    }
    if (diag != 0.0)
    {
      out.set(row, row, diag);
    }
  }
}

void TimeDirichletControlResidual::eliminateStateColumns(
    MatrixBuilder& J,
    Vector<Real>&  rhs) const
{
  if (rhs.size() != dims_.num_residuals)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual RHS size mismatch");
  }
  if (J.numRows() != dims_.num_residuals || J.numCols() != dims_.num_states)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual matrix size mismatch");
  }

  const Vector<Index> rows = constrainedRows();
  Vector<char>        is_constrained(dims_.num_states, 0);
  for (Index row : rows)
  {
    if (row < 0 || row >= dims_.num_states)
    {
      throw std::runtime_error(
          "TimeDirichletControlResidual constrained row is out of range");
    }
    is_constrained[row] = 1;
  }

  auto* sparse = dynamic_cast<CsrAssemblyMatrix*>(&J);
  if (sparse == nullptr)
  {
    return;
  }

  CsrMatrix&   mat  = sparse->mat();
  const Index* rp   = mat.rowPtrData();
  const Index* ci   = mat.colIndData();
  Real*        vals = mat.valuesData();

  for (Index row = 0; row < mat.rows(); ++row)
  {
    if (is_constrained[row] != 0)
    {
      continue;
    }
    for (Index k = rp[row]; k < rp[row + 1]; ++k)
    {
      const Index col = ci[k];
      if (is_constrained[col] != 0)
      {
        rhs[row] -= vals[k] * rhs[col];
        vals[k]   = 0.0;
      }
    }
  }
}

void TimeDirichletControlResidual::applyControlParamJac(
    const TimeContext&  ctx,
    const Vector<Real>& dir,
    Vector<Real>&       out) const
{
  if (dir.size() != dims_.num_params)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual parameter direction size mismatch");
  }
  resizeOrZero(out, dims_.num_residuals);
  const LinearInterpolation&  interp = ctr_time_stencils_[ctx.step];
  BlockVectorView<const Real> params(
      dir.data() + ctr_param_offset_,
      ctr_param_levels_,
      ctr_.numDofs());
  for (Index i = 0; i < ctr_.numDofs(); ++i)
  {
    Real value = 0.0;
    interp.forEachWeight(
        [&](Index level, Real wt)
        {
          value += wt * params(level, i);
        });
    out[ctr_.stateDof(i)] -= value;
  }
}

void TimeDirichletControlResidual::applyControlParamJacT(
    const TimeContext&  ctx,
    const Vector<Real>& adj,
    Vector<Real>&       out) const
{
  if (adj.size() != dims_.num_residuals)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual adjoint vector size mismatch");
  }
  resizeOrZero(out, dims_.num_params);
  const LinearInterpolation& interp = ctr_time_stencils_[ctx.step];
  BlockVectorView<Real>      params(
      out.data() + ctr_param_offset_,
      ctr_param_levels_,
      ctr_.numDofs());
  for (Index i = 0; i < ctr_.numDofs(); ++i)
  {
    const Real value = adj[ctr_.stateDof(i)];
    interp.forEachWeight(
        [&](Index level, Real wt)
        {
          params(level, i) -= wt * value;
        });
  }
}

void TimeDirichletControlResidual::initializeControlTimeStencils(
    Vector<LinearInterpolation> ctr_time_stencils)
{
  if (ctr_time_stencils.empty())
  {
    ctr_time_stencils_.resize(base_dims_.num_steps);
    for (Index step = 0; step < base_dims_.num_steps; ++step)
    {
      ctr_time_stencils_[step] = {step, step, 0.0};
    }
    ctr_param_levels_ = base_dims_.num_steps;
    return;
  }
  if (ctr_time_stencils.size() != base_dims_.num_steps)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual control time stencil count mismatch");
  }

  ctr_time_stencils_ = std::move(ctr_time_stencils);
  ctr_param_levels_  = 0;
  for (const LinearInterpolation& stencil : ctr_time_stencils_)
  {
    if (!stencil.isValid())
    {
      throw std::runtime_error(
          "TimeDirichletControlResidual received invalid control time stencil");
    }
    ctr_param_levels_ =
        std::max(ctr_param_levels_, stencil.upper + 1);
  }
  if (ctr_param_levels_ <= 0 && ctr_.numDofs() > 0)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual has no control time levels");
  }
}

Real TimeDirichletControlResidual::ctrValue(
    Index               step,
    Index               i,
    const Vector<Real>& prm) const
{
  const LinearInterpolation&  interp = ctr_time_stencils_[step];
  Real                        value  = 0.0;
  BlockVectorView<const Real> params(
      prm.data() + ctr_param_offset_,
      ctr_param_levels_,
      ctr_.numDofs());
  interp.forEachWeight(
      [&](Index level, Real wt)
      {
        value += wt * params(level, i);
      });
  return value;
}

Index TimeDirichletControlResidual::ctrIndex(Index level, Index i) const
{
  return ctr_param_offset_ + level * ctr_.numDofs() + i;
}

Real TimeDirichletControlResidual::fixedValue(Index step, Index i) const
{
  if (fvals_.size() == fdofs_.size())
  {
    return fvals_[i];
  }
  BlockVectorView<const Real> values(
      fvals_.data(), base_dims_.num_steps, fdofs_.size());
  return values(step, i);
}

Vector<Index> TimeDirichletControlResidual::constrainedRows() const
{
  Vector<Index> rows;
  rows.reserve(ctr_.numDofs() + fdofs_.size());
  for (Index row : ctr_.stateDofs())
  {
    rows.push_back(row);
  }
  for (Index row : fdofs_)
  {
    rows.push_back(row);
  }
  return rows;
}

} // namespace assembly
} // namespace femx
