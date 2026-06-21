#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/assembly/TimeDirichletControlResidual.hpp>
#include <femx/linalg/SparseMatrix.hpp>
#include <femx/linalg/native/SparseMatrixOperator.hpp>

#if defined(FEMX_HAS_PETSC)
#include <femx/linalg/petsc/PETScMatrixOperator.hpp>
#endif

namespace femx
{
namespace assembly
{

namespace
{

void resize(Vector<Real>& out, Index size)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    out.setZero();
  }
}

void replaceSparseRow(linalg::SparseMatrixOperator& mat,
                      Index                         row,
                      Real                          diag)
{
  SparseMatrix& sparse = mat.matrix();
  if (row < 0 || row >= sparse.rows())
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual sparse row is out of range");
  }

  const Index* row_ptr = sparse.rowPtrData();
  const Index* col_ind = sparse.colIndData();
  Real*        values  = sparse.valuesData();

  bool has_diag = false;
  for (Index k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
  {
    values[k] = 0.0;
    if (col_ind[k] == row)
    {
      values[k] = diag;
      has_diag  = true;
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
    const problem::TimeResidual& base,
    DirichletControl             control,
    Vector<Index>                fixed_dofs,
    Index                        ctr_param_offset,
    Index                        num_params,
    Vector<Real>                 fixed_values,
    Vector<LinearInterpolation>  ctr_time_stencils)
  : base_(base),
    ctr_(std::move(control)),
    fixed_dofs_(std::move(fixed_dofs)),
    fixed_values_(std::move(fixed_values)),
    base_dims_(base.dimensions()),
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

  if (fixed_values_.empty())
  {
    fixed_values_.resize(fixed_dofs_.size());
  }
  else if (fixed_values_.size() != fixed_dofs_.size()
           && fixed_values_.size()
                  != base_dims_.num_steps * fixed_dofs_.size())
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual fixed value size mismatch");
  }

  for (Index dof : ctr_.stateDofs())
  {
    if (dof < 0 || dof >= base_dims_.num_states)
    {
      throw std::runtime_error(
          "TimeDirichletControlResidual control dof is out of range");
    }
  }
  for (Index dof : fixed_dofs_)
  {
    if (dof < 0 || dof >= base_dims_.num_states)
    {
      throw std::runtime_error(
          "TimeDirichletControlResidual fixed dof is out of range");
    }
    for (Index ctr_dof : ctr_.stateDofs())
    {
      if (dof == ctr_dof)
      {
        throw std::runtime_error(
            "TimeDirichletControlResidual received overlapping dofs");
      }
    }
  }
}

problem::TimeDims TimeDirichletControlResidual::dimensions() const
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

void TimeDirichletControlResidual::residual(
    const problem::TimeContext& ctx,
    Vector<Real>&               out) const
{
  checkContext(ctx);

  problem::TimeContext base_ctx = ctx;
  base_ctx.prm                  = &base_prm_;
  base_.residual(base_ctx, out);
  if (out.size() != dims_.num_residuals)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual base residual size mismatch");
  }

  for (Index i = 0; i < ctr_.numDofs(); ++i)
  {
    const Index row = ctr_.stateDof(i);
    out[row]        = (*ctx.next_state)[row] - ctrValue(ctx.step, i, *ctx.prm);
  }
  for (Index i = 0; i < fixed_dofs_.size(); ++i)
  {
    const Index row = fixed_dofs_[i];
    out[row]        = (*ctx.next_state)[row] - fixedValue(ctx.step, i);
  }
}

void TimeDirichletControlResidual::applyJac(
    const problem::TimeContext& ctx,
    problem::VariableBlock      wrt,
    const Vector<Real>&         dir,
    Vector<Real>&               out) const
{
  checkContext(ctx);
  if (wrt == problem::VariableBlock::Param)
  {
    applyControlParamJac(ctx, dir, out);
    return;
  }

  problem::TimeContext base_ctx = ctx;
  base_ctx.prm                  = &base_prm_;
  base_.applyJac(base_ctx, wrt, dir, out);
  if (out.size() != dims_.num_residuals)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual Jacobian apply size mismatch");
  }

  const Real diag =
      wrt == problem::VariableBlock::NextState ? 1.0 : 0.0;
  for (Index row : constrainedRows())
  {
    out[row] = diag == 0.0 ? 0.0 : dir[row];
  }
}

void TimeDirichletControlResidual::applyJacT(
    const problem::TimeContext& ctx,
    problem::VariableBlock      wrt,
    const Vector<Real>&         adjoint,
    Vector<Real>&               out) const
{
  checkContext(ctx);
  if (wrt == problem::VariableBlock::Param)
  {
    applyControlParamJacT(ctx, adjoint, out);
    return;
  }

  throw std::runtime_error(
      "TimeDirichletControlResidual state transpose apply requires assembled Jacobians");
}

bool TimeDirichletControlResidual::assembleJacobian(
    const problem::TimeContext& ctx,
    problem::VariableBlock      wrt,
    linalg::MatrixBuilder&      out) const
{
  checkContext(ctx);
  if (wrt == problem::VariableBlock::Param)
  {
    out.resize(dims_.num_residuals, dims_.num_params);
    out.setZero();
    const LinearInterpolation& interp = ctr_time_stencils_[ctx.step];
    for (Index i = 0; i < ctr_.numDofs(); ++i)
    {
      const Index row = ctr_.stateDof(i);
      interp.forEachWeight(
          [&](Index level, Real weight)
          {
            out.set(row, ctrIndex(level, i), -weight);
          });
    }
    return true;
  }

  problem::TimeContext base_ctx = ctx;
  base_ctx.prm                  = &base_prm_;
  if (!base_.assembleJacobian(base_ctx, wrt, out))
  {
    return false;
  }

  replaceStateRows(out, wrt == problem::VariableBlock::NextState ? 1.0 : 0.0);
  return true;
}

const DirichletControl& TimeDirichletControlResidual::control() const
{
  return ctr_;
}

void TimeDirichletControlResidual::checkContext(
    const problem::TimeContext& ctx) const
{
  if (ctx.step < 0 || ctx.step >= dims_.num_steps)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual step is out of range");
  }
  checkVector(ctx.prev_state, dims_.num_states, "previous state");
  checkVector(ctx.next_state, dims_.num_states, "next state");
  checkVector(ctx.prm, dims_.num_params, "parameter");
}

void TimeDirichletControlResidual::checkVector(const Vector<Real>* value,
                                               Index               size,
                                               const char*         name) const
{
  if (value == nullptr || value->size() != size)
  {
    throw std::runtime_error(std::string("TimeDirichletControlResidual ")
                             + name + " size mismatch");
  }
}

void TimeDirichletControlResidual::replaceStateRows(
    linalg::MatrixBuilder& out,
    Real                   diag) const
{
  if (out.numRows() != dims_.num_residuals
      || out.numCols() != dims_.num_states)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual matrix size mismatch");
  }

  const Vector<Index> rows = constrainedRows();
  if (auto* sparse = dynamic_cast<linalg::SparseMatrixOperator*>(&out))
  {
    for (Index row : rows)
    {
      replaceSparseRow(*sparse, row, diag);
    }
    return;
  }

#if defined(FEMX_HAS_PETSC)
  if (auto* petsc = dynamic_cast<linalg::PETScMatrixOperator*>(&out))
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

void TimeDirichletControlResidual::applyControlParamJac(
    const problem::TimeContext& ctx,
    const Vector<Real>&         dir,
    Vector<Real>&               out) const
{
  if (dir.size() != dims_.num_params)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual parameter direction size mismatch");
  }
  resize(out, dims_.num_residuals);
  const LinearInterpolation& interp = ctr_time_stencils_[ctx.step];
  for (Index i = 0; i < ctr_.numDofs(); ++i)
  {
    Real value = 0.0;
    interp.forEachWeight(
        [&](Index level, Real weight)
        {
          value += weight * dir[ctrIndex(level, i)];
        });
    out[ctr_.stateDof(i)] -= value;
  }
}

void TimeDirichletControlResidual::applyControlParamJacT(
    const problem::TimeContext& ctx,
    const Vector<Real>&         adjoint,
    Vector<Real>&               out) const
{
  if (adjoint.size() != dims_.num_residuals)
  {
    throw std::runtime_error(
        "TimeDirichletControlResidual adjoint vector size mismatch");
  }
  resize(out, dims_.num_params);
  const LinearInterpolation& interp = ctr_time_stencils_[ctx.step];
  for (Index i = 0; i < ctr_.numDofs(); ++i)
  {
    const Real value = adjoint[ctr_.stateDof(i)];
    interp.forEachWeight(
        [&](Index level, Real weight)
        {
          out[ctrIndex(level, i)] -= weight * value;
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
  const LinearInterpolation& interp = ctr_time_stencils_[step];
  Real                       value  = 0.0;
  interp.forEachWeight(
      [&](Index level, Real weight)
      {
        value += weight * prm[ctrIndex(level, i)];
      });
  return value;
}

Index TimeDirichletControlResidual::ctrIndex(Index level, Index i) const
{
  return ctr_param_offset_ + level * ctr_.numDofs() + i;
}

Real TimeDirichletControlResidual::fixedValue(Index step, Index i) const
{
  if (fixed_values_.size() == fixed_dofs_.size())
  {
    return fixed_values_[i];
  }
  return fixed_values_[step * fixed_dofs_.size() + i];
}

Vector<Index> TimeDirichletControlResidual::constrainedRows() const
{
  Vector<Index> rows;
  rows.reserve(ctr_.numDofs() + fixed_dofs_.size());
  for (Index row : ctr_.stateDofs())
  {
    rows.push_back(row);
  }
  for (Index row : fixed_dofs_)
  {
    rows.push_back(row);
  }
  return rows;
}

} // namespace assembly
} // namespace femx
