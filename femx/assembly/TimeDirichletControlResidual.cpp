#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/assembly/TimeDirichletControlResidual.hpp>
#include <femx/linalg/BlockVectorView.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/linalg/native/MapCsrMatrix.hpp>

#if defined(FEMX_HAS_PETSC)
#include <femx/linalg/petsc/PETScOperator.hpp>
#endif
using namespace femx::state;
using namespace femx::linalg;

namespace femx
{
namespace assembly
{

namespace
{

void replaceSparseRow(MapCsrMatrix& mat,
                      Index         row,
                      Real          diag)
{
  HostCsrMatrix& sparse = mat.mat();
  if (row < 0 || row >= sparse.rows())
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual sparse row is out of range");
  }

  const Index* rp   = sparse.rowPtrData();
  const Index* ci   = sparse.colIndData();
  Real*        vals = sparse.valsData();

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
    const TimeResidual&        base,
    fem::DirichletControl      ctr,
    Array<Index>               fdofs,
    Index                      ctr_param_offset,
    Index                      num_param,
    HostVector                 fvals,
    Array<LinearInterpolation> ctr_time_stencils)
  : base_(base),
    ctr_(std::move(ctr)),
    fdofs_(std::move(fdofs)),
    fvals_(std::move(fvals)),
    base_dims_(base.dims()),
    dims_(base_dims_),
    ctr_param_offset_(ctr_param_offset)
{
  if (base_dims_.num_res != base_dims_.num_states)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual requires square state residuals");
  }
  if (base_dims_.num_param != 0)
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

  const Index required_ctr_params =
      ctr_param_levels_ * ctr_.numControlParams();
  dims_.num_param =
      num_param < 0
          ? ctr_param_offset_ + required_ctr_params
          : num_param;
  if (dims_.num_param < ctr_param_offset_ + required_ctr_params)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual parameter count is too small");
  }
  base_prm_.resize(base_dims_.num_param);

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
  return dims_.num_param;
}

Index TimeDirichletControlResidual::numRes() const
{
  return dims_.num_res;
}

void TimeDirichletControlResidual::res(
    const TimeContext& ctx,
    HostVector&        out) const
{
  checkContext(ctx);

  TimeContext base_ctx = ctx;
  base_ctx.prm         = &base_prm_;
  base_.res(base_ctx, out);
  if (out.size() != dims_.num_res)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual base residual size mismatch");
  }

  const HostVector control = interpolatedControl(ctx.step, *ctx.prm);
  HostVector       vals;
  ctr_.apply(control, vals);
  for (Index i = 0; i < ctr_.numStateDofs(); ++i)
  {
    const Index row = ctr_.stateDof(i);
    out[row]        = (*ctx.nxt)[row] - vals[i];
  }
  for (Index i = 0; i < fdofs_.size(); ++i)
  {
    const Index row = fdofs_[i];
    out[row]        = (*ctx.nxt)[row] - fixedValue(ctx.step, i);
  }
}

void TimeDirichletControlResidual::applyJac(
    const TimeContext& ctx,
    VariableBlock      wrt,
    const HostVector&  dir,
    HostVector&        out) const
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
  if (out.size() != dims_.num_res)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual Jacobian apply size mismatch");
  }

  const Real diag = wrt.isNextState() ? 1.0 : 0.0;
  for (Index row : bcRows())
  {
    out[row] = diag == 0.0 ? 0.0 : dir[row];
  }
}

void TimeDirichletControlResidual::applyJacT(
    const TimeContext& ctx,
    VariableBlock      wrt,
    const HostVector&  adj,
    HostVector&        out) const
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
    MatrixOperator&    out) const
{
  checkContext(ctx);
  if (wrt.isParam())
  {
    out.resize(dims_.num_res, dims_.num_param);
    out.setZero();
    const LinearInterpolation& interp = ctr_time_stencils_[ctx.step];
    for (const fem::DirichletControlMapEntry& entry : ctr_.mapEntries())
    {
      interp.forEachWeight(
          [&](Index level, Real wt)
          {
            out.set(ctr_.stateDof(entry.state_row),
                    ctrIndex(level, entry.ctr_col),
                    -wt * entry.weight);
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
    MatrixOperator&    J,
    HostVector&        rhs) const
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
  checkVector(ctx.prm, dims_.num_param);
}

void TimeDirichletControlResidual::checkVector(const HostVector* value,
                                               Index             size) const
{
  if (value == nullptr || value->size() != size)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual vector size mismatch");
  }
}

void TimeDirichletControlResidual::replaceStateRows(
    MatrixOperator& out,
    Real            diag) const
{
  if (out.numRows() != dims_.num_res
      || out.numCols() != dims_.num_states)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual matrix size mismatch");
  }

  const Array<Index> rows = bcRows();
  if (auto* sparse = dynamic_cast<MapCsrMatrix*>(&out))
  {
    for (Index row : rows)
    {
      replaceSparseRow(*sparse, row, diag);
    }
    return;
  }

#if defined(FEMX_HAS_PETSC)
  if (auto* petsc = dynamic_cast<PETScOperator*>(&out))
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
    MatrixOperator& J,
    HostVector&     rhs) const
{
  if (rhs.size() != dims_.num_res)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual RHS size mismatch");
  }
  if (J.numRows() != dims_.num_res || J.numCols() != dims_.num_states)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual matrix size mismatch");
  }

  const Array<Index> rows = bcRows();
  Array<char>        is_constrained(dims_.num_states, 0);
  for (Index row : rows)
  {
    if (row < 0 || row >= dims_.num_states)
    {
      throw std::runtime_error(
          "TimeDirichletControlResidual constrained row is out of range");
    }
    is_constrained[row] = 1;
  }

  auto* sparse = dynamic_cast<MapCsrMatrix*>(&J);
  if (sparse == nullptr)
  {
    return;
  }

  HostCsrMatrix& mat  = sparse->mat();
  const Index*   rp   = mat.rowPtrData();
  const Index*   ci   = mat.colIndData();
  Real*          vals = mat.valsData();

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
    const TimeContext& ctx,
    const HostVector&  dir,
    HostVector&        out) const
{
  if (dir.size() != dims_.num_param)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual parameter direction size mismatch");
  }
  resizeOrZero(out, dims_.num_res);
  const HostVector control = interpolatedControl(ctx.step, dir);
  HostVector       mapped;
  ctr_.apply(control, mapped);
  for (Index i = 0; i < ctr_.numStateDofs(); ++i)
  {
    out[ctr_.stateDof(i)] -= mapped[i];
  }
}

void TimeDirichletControlResidual::applyControlParamJacT(
    const TimeContext& ctx,
    const HostVector&  adj,
    HostVector&        out) const
{
  if (adj.size() != dims_.num_res)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual adjoint vector size mismatch");
  }
  resizeOrZero(out, dims_.num_param);
  const LinearInterpolation& interp = ctr_time_stencils_[ctx.step];
  HostVector                 state_adj(ctr_.numStateDofs());
  for (Index i = 0; i < ctr_.numStateDofs(); ++i)
  {
    state_adj[i] = adj[ctr_.stateDof(i)];
  }

  HostVector ctr_adj;
  ctr_.applyTranspose(state_adj, ctr_adj);
  for (Index i = 0; i < ctr_.numControlParams(); ++i)
  {
    interp.forEachWeight(
        [&](Index level, Real wt)
        {
          out[ctrIndex(level, i)] -= wt * ctr_adj[i];
        });
  }
}

void TimeDirichletControlResidual::initializeControlTimeStencils(
    Array<LinearInterpolation> ctr_time_stencils)
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
  if (ctr_param_levels_ <= 0 && ctr_.numControlParams() > 0)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual has no control time levels");
  }
}

HostVector TimeDirichletControlResidual::interpolatedControl(
    Index             step,
    const HostVector& prm) const
{
  HostVector out(ctr_.numControlParams());
  if (out.empty())
  {
    return out;
  }

  const LinearInterpolation&  interp = ctr_time_stencils_[step];
  BlockVectorView<const Real> params(
      prm.data() + ctr_param_offset_,
      ctr_param_levels_,
      ctr_.numControlParams());
  for (Index i = 0; i < ctr_.numControlParams(); ++i)
  {
    interp.forEachWeight(
        [&](Index level, Real wt)
        {
          out[i] += wt * params(level, i);
        });
  }
  return out;
}

Index TimeDirichletControlResidual::ctrIndex(Index level, Index i) const
{
  return ctr_param_offset_ + level * ctr_.numControlParams() + i;
}

Real TimeDirichletControlResidual::fixedValue(Index step, Index i) const
{
  if (fvals_.size() == fdofs_.size())
  {
    return fvals_[i];
  }
  BlockVectorView<const Real> vals(
      fvals_.data(), base_dims_.num_steps, fdofs_.size());
  return vals(step, i);
}

Array<Index> TimeDirichletControlResidual::bcRows() const
{
  Array<Index> rows;
  rows.reserve(ctr_.numStateDofs() + fdofs_.size());
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
