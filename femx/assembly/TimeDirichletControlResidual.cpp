#include <stdexcept>
#include <utility>

#include <femx/assembly/TimeDirichletControlResidual.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/linalg/native/MapCsrMatrix.hpp>

#if defined(FEMX_HAS_PETSC)
#include <femx/linalg/petsc/PETScOperator.hpp>
#endif

namespace femx
{
namespace assembly
{
namespace
{

void replaceSparseRow(linalg::MapCsrMatrix& mat,
                      Index                 row,
                      Real                  diag)
{
  HostCsrMatrix& csr = mat.mat();
  if (row < 0 || row >= csr.rows())
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual sparse row is out of range");
  }

  bool has_diag = false;
  for (Index k = csr.rowPtrData()[row];
       k < csr.rowPtrData()[row + 1];
       ++k)
  {
    csr.valsData()[k] = 0.0;
    if (csr.colIndData()[k] == row)
    {
      csr.valsData()[k] = diag;
      has_diag          = true;
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
    const state::TimeResidual& base,
    fem::DirichletControl      ctr,
    Array<Index>               fixed_dofs,
    Index                      ctr_off,
    Index                      num_prm,
    HostVector                 fixed_vals,
    Array<LinearInterpolation> time)
  : base_(base),
    base_dims_(base.dims()),
    dims_(base_dims_)
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

  ctr_            = fem::makeControlMap(base_dims_.num_steps,
                             base_dims_.num_states,
                             ctr,
                             std::move(fixed_dofs),
                             std::move(fixed_vals),
                             std::move(time),
                             ctr_off,
                             num_prm);
  dims_.num_param = ctr_.numParams();
  rows_.resize(ctr_.numBcs());
  for (Index ib = 0; ib < rows_.size(); ++ib)
  {
    rows_[ib] = ctr_.dofs()[ib];
  }
  base_prm_.resize(base_dims_.num_param);
  bc_vals_.resize(ctr_.numBcs());
}

state::TimeDims TimeDirichletControlResidual::dims() const
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

void TimeDirichletControlResidual::res(const state::TimeContext& ctx,
                                       HostVector&               out) const
{
  checkContext(ctx);

  state::TimeContext base_ctx = ctx;
  base_ctx.prm                = &base_prm_;
  base_.res(base_ctx, out);
  if (out.size() != dims_.num_res)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual base residual size mismatch");
  }

  fem::controlVals(
      ctr_.view(), ctx.step, ctx.prm->view(), bc_vals_.view());
  for (Index ib = 0; ib < rows_.size(); ++ib)
  {
    const Index row = rows_[ib];
    out[row]        = (*ctx.nxt)[row] - bc_vals_[ib];
  }
}

void TimeDirichletControlResidual::applyJac(
    const state::TimeContext& ctx,
    state::VariableBlock      wrt,
    const HostVector&         dir,
    HostVector&               out) const
{
  checkContext(ctx);
  if (wrt.isParam())
  {
    resizeOrZero(out, dims_.num_res);
    fem::controlJac(ctr_.view(), ctx.step, dir.view(), out.view());
    return;
  }

  state::TimeContext base_ctx = ctx;
  base_ctx.prm                = &base_prm_;
  base_.applyJac(base_ctx, wrt, dir, out);
  if (out.size() != dims_.num_res)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual Jacobian apply size mismatch");
  }

  const Real diag = wrt.isNextState() ? 1.0 : 0.0;
  for (Index row : rows_)
  {
    out[row] = diag * dir[row];
  }
}

void TimeDirichletControlResidual::applyJacT(
    const state::TimeContext& ctx,
    state::VariableBlock      wrt,
    const HostVector&         adj,
    HostVector&               out) const
{
  checkContext(ctx);
  if (!wrt.isParam())
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual state transpose apply requires assembled Jacobians");
  }

  resizeOrZero(out, dims_.num_param);
  fem::addControlJacT(
      ctr_.view(), ctx.step, adj.view(), out.view());
}

bool TimeDirichletControlResidual::assembleJac(
    const state::TimeContext& ctx,
    state::VariableBlock      wrt,
    linalg::MatrixOperator&   out) const
{
  checkContext(ctx);
  if (wrt.isParam())
  {
    out.resize(dims_.num_res, dims_.num_param);
    out.setZero();
    const auto  map  = ctr_.view();
    const Index lo   = map.lowerLevel(ctx.step);
    const Index hi   = map.upperLevel(ctx.step);
    const Real  hi_w = map.upperWt(ctx.step);
    const Real  lo_w = 1.0 - hi_w;
    for (Index i = 0; i < map.num_ctr; ++i)
    {
      const Index row = map.dof(i);
      for (Index k = map.ctrBegin(i); k < map.ctrEnd(i); ++k)
      {
        const Index col = map.ctrCol(k);
        const Real  val = -map.ctrWt(k);
        if (lo_w != 0.0)
        {
          out.set(row, map.ctrIndex(lo, col), lo_w * val);
        }
        if (hi != lo && hi_w != 0.0)
        {
          out.set(row, map.ctrIndex(hi, col), hi_w * val);
        }
      }
    }
    return true;
  }

  state::TimeContext base_ctx = ctx;
  base_ctx.prm                = &base_prm_;
  if (!base_.assembleJac(base_ctx, wrt, out))
  {
    return false;
  }
  replaceRows(out, wrt.isNextState() ? 1.0 : 0.0);
  return true;
}

void TimeDirichletControlResidual::prepareLinearSolve(
    const state::TimeContext& ctx,
    state::VariableBlock      wrt,
    linalg::MatrixOperator&   jac,
    HostVector&               rhs) const
{
  checkContext(ctx);
  if (wrt.isNextState())
  {
    eliminateCols(jac, rhs);
  }
}

void TimeDirichletControlResidual::checkContext(
    const state::TimeContext& ctx) const
{
  if (ctx.step < 0 || ctx.step >= dims_.num_steps)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual step is out of range");
  }
  if (ctx.hist.count() < dims_.num_hist
      || ctx.hist.stateSize() != dims_.num_states)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual history state size mismatch");
  }
  if (ctx.nxt == nullptr || ctx.nxt->size() != dims_.num_states
      || ctx.prm == nullptr || ctx.prm->size() != dims_.num_param)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual vector size mismatch");
  }
}

void TimeDirichletControlResidual::replaceRows(
    linalg::MatrixOperator& mat,
    Real                    diag) const
{
  if (mat.numRows() != dims_.num_res
      || mat.numCols() != dims_.num_states)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual matrix size mismatch");
  }

  if (auto* sparse = dynamic_cast<linalg::MapCsrMatrix*>(&mat))
  {
    for (Index row : rows_)
    {
      replaceSparseRow(*sparse, row, diag);
    }
    return;
  }

#if defined(FEMX_HAS_PETSC)
  if (auto* petsc = dynamic_cast<linalg::PETScOperator*>(&mat))
  {
    mat.finalize();
    petsc->zeroRows(rows_, diag);
    return;
  }
#endif

  for (Index row : rows_)
  {
    for (Index col = 0; col < mat.numCols(); ++col)
    {
      mat.set(row, col, 0.0);
    }
    if (diag != 0.0)
    {
      mat.set(row, row, diag);
    }
  }
}

void TimeDirichletControlResidual::eliminateCols(
    linalg::MatrixOperator& mat_op,
    HostVector&             rhs) const
{
  if (rhs.size() != dims_.num_res
      || mat_op.numRows() != dims_.num_res
      || mat_op.numCols() != dims_.num_states)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual linear system size mismatch");
  }

  auto* sparse = dynamic_cast<linalg::MapCsrMatrix*>(&mat_op);
  if (sparse == nullptr)
  {
    return;
  }

  Array<char> fixed(dims_.num_states, 0);
  for (Index row : rows_)
  {
    fixed[row] = 1;
  }

  HostCsrMatrix& mat = sparse->mat();
  for (Index row = 0; row < mat.rows(); ++row)
  {
    if (fixed[row] != 0)
    {
      continue;
    }
    for (Index k = mat.rowPtrData()[row];
         k < mat.rowPtrData()[row + 1];
         ++k)
    {
      const Index col = mat.colIndData()[k];
      if (fixed[col] != 0)
      {
        rhs[row]          -= mat.valsData()[k] * rhs[col];
        mat.valsData()[k]  = 0.0;
      }
    }
  }
}

} // namespace assembly
} // namespace femx
