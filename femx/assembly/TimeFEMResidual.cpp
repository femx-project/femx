#include <stdexcept>
#include <string>
#include <utility>

#include <femx/assembly/Assembler.hpp>
#include <femx/assembly/TimeFEMResidual.hpp>

#if defined(FEMX_HAS_PETSC)
#include <petscsys.h>
#endif

namespace femx
{
namespace assembly
{

namespace
{

#if defined(FEMX_HAS_PETSC)
void allreduce(Vector<Real>& out)
{
  const int ierr = MPI_Allreduce(MPI_IN_PLACE,
                                 out.data(),
                                 static_cast<int>(out.size()),
                                 MPIU_REAL,
                                 MPI_SUM,
                                 PETSC_COMM_WORLD);
  if (ierr != MPI_SUCCESS)
  {
    throw std::runtime_error("TimeFEMResidual MPI_Allreduce failed");
  }
}
#endif

} // namespace

TimeFEMResidual::TimeFEMResidual(Index                    num_steps,
                                 DofLayout                res_layout,
                                 DofLayout                state_layout,
                                 const TimeElementKernel& kernel)
  : TimeFEMResidual(num_steps,
                    std::move(res_layout),
                    state_layout,
                    std::move(state_layout),
                    kernel)
{
}

TimeFEMResidual::TimeFEMResidual(Index                    num_steps,
                                 DofLayout                res_layout,
                                 DofLayout                prev_state_layout,
                                 DofLayout                next_state_layout,
                                 const TimeElementKernel& kernel)
  : num_steps_(num_steps),
    res_layout_(std::move(res_layout)),
    prev_state_layout_(std::move(prev_state_layout)),
    next_state_layout_(std::move(next_state_layout)),
    kernel_(kernel),
    cell_end_(res_layout_.numElems())
{
  checkLayouts();
}

TimeFEMResidual::TimeFEMResidual(Index                    num_steps,
                                 DofLayout                res_layout,
                                 DofLayout                prev_state_layout,
                                 DofLayout                next_state_layout,
                                 DofLayout                param_layout,
                                 const TimeElementKernel& kernel)
  : num_steps_(num_steps),
    res_layout_(std::move(res_layout)),
    prev_state_layout_(std::move(prev_state_layout)),
    next_state_layout_(std::move(next_state_layout)),
    param_layout_(std::move(param_layout)),
    kernel_(kernel),
    cell_end_(res_layout_.numElems())
{
  checkLayouts();
}

void TimeFEMResidual::setCellRange(Index begin, Index end)
{
  if (begin < 0 || end < begin || end > numCells())
  {
    throw std::runtime_error("TimeFEMResidual received invalid cell range");
  }
#if !defined(FEMX_HAS_PETSC)
  if (begin != 0 || end != numCells())
  {
    throw std::runtime_error("TimeFEMResidual cell ranges require PETSc");
  }
#endif
  cell_begin_ = begin;
  cell_end_   = end;
}

problem::TimeDims TimeFEMResidual::dimensions() const
{
  return {num_steps_,
          next_state_layout_.numDofs(),
          numParams(),
          res_layout_.numDofs()};
}

void TimeFEMResidual::residual(const problem::TimeContext& ctx,
                               Vector<Real>&               out) const
{
  checkContext(ctx);

  Assembler assembler(res_layout_);
  assembler.initVector(out);

  Vector<Real> prev_e;
  Vector<Real> next_e;
  Vector<Real> prm_e;
  Vector<Real> res_e;
  for (Index ic = cell_begin_; ic < cell_end_; ++ic)
  {
    gather(prev_state_layout_, *ctx.prev_state, ic, prev_e);
    gather(next_state_layout_, *ctx.next_state, ic, next_e);
    prm_e = gatherParam(ic, *ctx.prm);
    kernel_.res(ctx.step, ic, prev_e, next_e, prm_e, res_e);
    assembler.addVector(ic, res_e, out);
  }

#if defined(FEMX_HAS_PETSC)
  if (cell_begin_ != 0 || cell_end_ != numCells())
  {
    allreduce(out);
  }
#endif
}

void TimeFEMResidual::applyJac(const problem::TimeContext& ctx,
                                    problem::VariableBlock      wrt,
                                    const Vector<Real>&         dir,
                                    Vector<Real>&               out) const
{
  checkContext(ctx);
  checkDirection(wrt, dir);
  resize(out, dimensions().num_residuals);

  const DofLayout* col_layout = layoutFor(wrt);
  if (col_layout == nullptr)
  {
    return;
  }

  Assembler    assembler(res_layout_);
  Vector<Real> prev_e;
  Vector<Real> next_e;
  Vector<Real> prm_e;
  Vector<Real> dir_e;
  Vector<Real> res_e;
  DenseMatrix  jac_e;
  for (Index ic = cell_begin_; ic < cell_end_; ++ic)
  {
    gather(prev_state_layout_, *ctx.prev_state, ic, prev_e);
    gather(next_state_layout_, *ctx.next_state, ic, next_e);
    prm_e = gatherParam(ic, *ctx.prm);
    gather(*col_layout, dir, ic, dir_e);
    kernel_.jacobian(ctx.step, ic, wrt, prev_e, next_e, prm_e, jac_e);
    matVec(jac_e, dir_e, res_e);
    assembler.addVector(ic, res_e, out);
  }

#if defined(FEMX_HAS_PETSC)
  if (cell_begin_ != 0 || cell_end_ != numCells())
  {
    allreduce(out);
  }
#endif
}

void TimeFEMResidual::applyJacT(const problem::TimeContext& ctx,
                                     problem::VariableBlock      wrt,
                                     const Vector<Real>&         adjoint,
                                     Vector<Real>&               out) const
{
  checkContext(ctx);
  if (adjoint.size() != dimensions().num_residuals)
  {
    throw std::runtime_error("TimeFEMResidual adjoint size mismatch");
  }

  const DofLayout* col_layout = layoutFor(wrt);
  if (col_layout == nullptr)
  {
    resize(out, numParams());
    return;
  }

  resize(out, col_layout->numDofs());

  Assembler    assembler(*col_layout);
  Vector<Real> prev_e;
  Vector<Real> next_e;
  Vector<Real> prm_e;
  Vector<Real> adj_e;
  Vector<Real> col_e;
  DenseMatrix  jac_e;
  for (Index ic = cell_begin_; ic < cell_end_; ++ic)
  {
    gather(prev_state_layout_, *ctx.prev_state, ic, prev_e);
    gather(next_state_layout_, *ctx.next_state, ic, next_e);
    prm_e = gatherParam(ic, *ctx.prm);
    gather(res_layout_, adjoint, ic, adj_e);
    kernel_.jacobian(ctx.step, ic, wrt, prev_e, next_e, prm_e, jac_e);
    matTVec(jac_e, adj_e, col_e);
    assembler.addVector(ic, col_e, out);
  }

#if defined(FEMX_HAS_PETSC)
  if (cell_begin_ != 0 || cell_end_ != numCells())
  {
    allreduce(out);
  }
#endif
}

bool TimeFEMResidual::assembleJacobian(const problem::TimeContext& ctx,
                                       problem::VariableBlock      wrt,
                                       linalg::MatrixBuilder&     out) const
{
  checkContext(ctx);
  const DofLayout* col_layout = layoutFor(wrt);
  if (col_layout == nullptr)
  {
    out.resize(dimensions().num_residuals, numParams());
    out.setZero();
    return true;
  }

  Assembler assembler(res_layout_, *col_layout);
  assembler.initMatrix(out);

  Vector<Real> prev_e;
  Vector<Real> next_e;
  Vector<Real> prm_e;
  DenseMatrix  jac_e;
  for (Index ic = cell_begin_; ic < cell_end_; ++ic)
  {
    gather(prev_state_layout_, *ctx.prev_state, ic, prev_e);
    gather(next_state_layout_, *ctx.next_state, ic, next_e);
    prm_e = gatherParam(ic, *ctx.prm);
    kernel_.jacobian(ctx.step, ic, wrt, prev_e, next_e, prm_e, jac_e);
    assembler.addMatrix(ic, jac_e, out);
  }
  return true;
}

Index TimeFEMResidual::numCells() const
{
  return res_layout_.numElems();
}

Index TimeFEMResidual::numParams() const
{
  return param_layout_ ? param_layout_->numDofs() : 0;
}

const DofLayout* TimeFEMResidual::layoutFor(problem::VariableBlock wrt) const
{
  if (wrt == problem::VariableBlock::PrevState)
  {
    return &prev_state_layout_;
  }
  if (wrt == problem::VariableBlock::NextState)
  {
    return &next_state_layout_;
  }
  return param_layout_ ? &*param_layout_ : nullptr;
}

void TimeFEMResidual::checkLayouts() const
{
  if (num_steps_ < 0)
  {
    throw std::runtime_error("TimeFEMResidual received negative step count");
  }
  if (res_layout_.numElems() != prev_state_layout_.numElems()
      || res_layout_.numElems() != next_state_layout_.numElems()
      || (param_layout_ && res_layout_.numElems() != param_layout_->numElems()))
  {
    throw std::runtime_error(
        "TimeFEMResidual layouts have different cell counts");
  }
  if (prev_state_layout_.numDofs() != next_state_layout_.numDofs())
  {
    throw std::runtime_error(
        "TimeFEMResidual previous and next state sizes differ");
  }
}

void TimeFEMResidual::checkContext(const problem::TimeContext& ctx) const
{
  const problem::TimeDims dims = dimensions();
  if (ctx.step < 0 || ctx.step >= dims.num_steps)
  {
    throw std::runtime_error("TimeFEMResidual step is out of range");
  }
  checkVector(ctx.prev_state, dims.num_states, "previous state");
  checkVector(ctx.next_state, dims.num_states, "next state");
  checkVector(ctx.prm, dims.num_params, "parameter");
}

void TimeFEMResidual::checkVector(const Vector<Real>* value,
                                  Index               size,
                                  const char*         name) const
{
  if (value == nullptr || value->size() != size)
  {
    throw std::runtime_error(std::string("TimeFEMResidual ") + name
                             + " size mismatch");
  }
}

void TimeFEMResidual::checkDirection(problem::VariableBlock wrt,
                                     const Vector<Real>&    dir) const
{
  const DofLayout* layout   = layoutFor(wrt);
  const Index      expected = layout == nullptr ? numParams() : layout->numDofs();
  if (dir.size() != expected)
  {
    throw std::runtime_error("TimeFEMResidual direction size mismatch");
  }
}

Vector<Real> TimeFEMResidual::gatherParam(Index               ic,
                                          const Vector<Real>& global) const
{
  Vector<Real> local;
  if (param_layout_)
  {
    gather(*param_layout_, global, ic, local);
  }
  return local;
}

void TimeFEMResidual::gather(const DofLayout&    layout,
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
    checkDof(dof, global.size());
    local[i] = global[dof];
  }
}

void TimeFEMResidual::matVec(const DenseMatrix&  mat,
                             const Vector<Real>& x,
                             Vector<Real>&       out)
{
  if (mat.cols() != x.size())
  {
    throw std::runtime_error("TimeFEMResidual local matrix size mismatch");
  }
  resize(out, mat.rows());
  for (Index i = 0; i < mat.rows(); ++i)
  {
    Real sum = 0.0;
    for (Index j = 0; j < mat.cols(); ++j)
    {
      sum += mat(i, j) * x[j];
    }
    out[i] = sum;
  }
}

void TimeFEMResidual::matTVec(const DenseMatrix&  mat,
                              const Vector<Real>& x,
                              Vector<Real>&       out)
{
  if (mat.rows() != x.size())
  {
    throw std::runtime_error("TimeFEMResidual local matrix size mismatch");
  }
  resize(out, mat.cols());
  for (Index j = 0; j < mat.cols(); ++j)
  {
    Real sum = 0.0;
    for (Index i = 0; i < mat.rows(); ++i)
    {
      sum += mat(i, j) * x[i];
    }
    out[j] = sum;
  }
}

void TimeFEMResidual::resize(Vector<Real>& out, Index size)
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

void TimeFEMResidual::checkDof(Index dof, Index size)
{
  if (dof < 0 || dof >= size)
  {
    throw std::runtime_error("TimeFEMResidual dof is out of range");
  }
}

} // namespace assembly
} // namespace femx
