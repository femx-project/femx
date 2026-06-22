#include <stdexcept>
#include <utility>

#include <femx/assembly/Assembler.hpp>
#include <femx/assembly/TimeFEMResidual.hpp>

#if defined(FEMX_HAS_PETSC)
#include <petscsys.h>
#endif

using namespace std;
using namespace femx::problem;
using namespace femx::linalg;

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
    throw runtime_error("TimeFEMResidual MPI_Allreduce failed");
  }
}
#endif

Vector<DofLayout> singleHistoryLayout(DofLayout lyt)
{
  Vector<DofLayout> out;
  out.push_back(std::move(lyt));
  return out;
}

} // namespace

TimeFEMResidual::TimeFEMResidual(Index                    nt,
                                 DofLayout                res_layout,
                                 DofLayout                state_layout,
                                 const TimeElementKernel& ker)
  : TimeFEMResidual(nt,
                    std::move(res_layout),
                    singleHistoryLayout(state_layout),
                    std::move(state_layout),
                    ker)
{
}

TimeFEMResidual::TimeFEMResidual(Index                    nt,
                                 DofLayout                res_layout,
                                 DofLayout                prev_state_layout,
                                 DofLayout                next_state_layout,
                                 const TimeElementKernel& ker)
  : TimeFEMResidual(nt,
                    std::move(res_layout),
                    singleHistoryLayout(std::move(prev_state_layout)),
                    std::move(next_state_layout),
                    ker)
{
}

TimeFEMResidual::TimeFEMResidual(
    Index                    nt,
    DofLayout                res_layout,
    Vector<DofLayout>        history_state_layouts,
    DofLayout                next_state_layout,
    const TimeElementKernel& ker)
  : nt_(nt),
    res_layout_(std::move(res_layout)),
    history_state_layouts_(std::move(history_state_layouts)),
    next_state_layout_(std::move(next_state_layout)),
    kernel_(ker),
    cell_end_(res_layout_.numElems())
{
  checkLayouts();
}

TimeFEMResidual::TimeFEMResidual(Index                    nt,
                                 DofLayout                res_layout,
                                 DofLayout                prev_state_layout,
                                 DofLayout                next_state_layout,
                                 DofLayout                param_layout,
                                 const TimeElementKernel& ker)
  : TimeFEMResidual(nt,
                    std::move(res_layout),
                    singleHistoryLayout(std::move(prev_state_layout)),
                    std::move(next_state_layout),
                    std::move(param_layout),
                    ker)
{
}

TimeFEMResidual::TimeFEMResidual(
    Index                    nt,
    DofLayout                res_layout,
    Vector<DofLayout>        history_state_layouts,
    DofLayout                next_state_layout,
    DofLayout                param_layout,
    const TimeElementKernel& ker)
  : nt_(nt),
    res_layout_(std::move(res_layout)),
    history_state_layouts_(std::move(history_state_layouts)),
    next_state_layout_(std::move(next_state_layout)),
    param_layout_(std::move(param_layout)),
    kernel_(ker),
    cell_end_(res_layout_.numElems())
{
  checkLayouts();
}

void TimeFEMResidual::setCellRange(Index begin, Index end)
{
  if (begin < 0 || end < begin || end > numCells())
  {
    throw runtime_error("TimeFEMResidual received invalid cell range");
  }
#if !defined(FEMX_HAS_PETSC)
  if (begin != 0 || end != numCells())
  {
    throw runtime_error("TimeFEMResidual cell ranges require PETSc");
  }
#endif
  cell_begin_ = begin;
  cell_end_   = end;
}

TimeDims TimeFEMResidual::dims() const
{
  return {nt_,
          next_state_layout_.numDofs(),
          numParams(),
          res_layout_.numDofs(),
          numHistoryStates()};
}

void TimeFEMResidual::res(const TimeContext& ctx,
                          Vector<Real>&      out) const
{
  checkContext(ctx);

  Assembler assembler(res_layout_, AssemblyMode::Atomic);
  assembler.initVec(out);

#pragma omp parallel
  {
    Vector<Real> hst_e;
    Vector<Real> next_e;
    Vector<Real> prm_e;
    Vector<Real> res_e;

#pragma omp for
    for (Index ic = cell_begin_; ic < cell_end_; ++ic)
    {
      gatherHistory(ctx, ic, hst_e);
      gather(next_state_layout_, *ctx.nxt, ic, next_e);
      prm_e = gatherParam(ic, *ctx.prm);
      kernel_.res(ctx.step,
                  ic,
                  TimeHistoryView(hst_e.data(),
                                  numHistoryStates(),
                                  next_state_layout_.numDofsPerElem()),
                  next_e,
                  prm_e,
                  res_e);
      assembler.addVec(ic, res_e, out);
    }
  }

#if defined(FEMX_HAS_PETSC)
  if (cell_begin_ != 0 || cell_end_ != numCells())
  {
    allreduce(out);
  }
#endif
}

void TimeFEMResidual::applyJac(const TimeContext&  ctx,
                               VariableBlock       wrt,
                               const Vector<Real>& dir,
                               Vector<Real>&       out) const
{
  checkContext(ctx);
  checkDirection(wrt, dir);
  resizeOrZero(out, dims().nres);

  const DofLayout* col_layout = layoutFor(wrt);
  if (col_layout == nullptr)
  {
    return;
  }

  Assembler assembler(res_layout_, AssemblyMode::Atomic);
#pragma omp parallel
  {
    Vector<Real> hst_e;
    Vector<Real> next_e;
    Vector<Real> prm_e;
    Vector<Real> dir_e;
    Vector<Real> res_e;
    DenseMatrix  jac_e;

#pragma omp for
    for (Index ic = cell_begin_; ic < cell_end_; ++ic)
    {
      gatherHistory(ctx, ic, hst_e);
      gather(next_state_layout_, *ctx.nxt, ic, next_e);
      prm_e = gatherParam(ic, *ctx.prm);
      gather(*col_layout, dir, ic, dir_e);
      kernel_.jacobian(ctx.step,
                       ic,
                       wrt,
                       TimeHistoryView(hst_e.data(),
                                       numHistoryStates(),
                                       next_state_layout_.numDofsPerElem()),
                       next_e,
                       prm_e,
                       jac_e);
      matVec(jac_e, dir_e, res_e);
      assembler.addVec(ic, res_e, out);
    }
  }

#if defined(FEMX_HAS_PETSC)
  if (cell_begin_ != 0 || cell_end_ != numCells())
  {
    allreduce(out);
  }
#endif
}

void TimeFEMResidual::applyJacT(const TimeContext&  ctx,
                                VariableBlock       wrt,
                                const Vector<Real>& adj,
                                Vector<Real>&       out) const
{
  checkContext(ctx);
  if (adj.size() != dims().nres)
  {
    throw runtime_error("TimeFEMResidual adjoint size mismatch");
  }

  const DofLayout* col_layout = layoutFor(wrt);
  if (col_layout == nullptr)
  {
    resizeOrZero(out, numParams());
    return;
  }

  resizeOrZero(out, col_layout->numDofs());

  Assembler assembler(*col_layout, AssemblyMode::Atomic);
#pragma omp parallel
  {
    Vector<Real> hst_e;
    Vector<Real> next_e;
    Vector<Real> prm_e;
    Vector<Real> adj_e;
    Vector<Real> col_e;
    DenseMatrix  jac_e;

#pragma omp for
    for (Index ic = cell_begin_; ic < cell_end_; ++ic)
    {
      gatherHistory(ctx, ic, hst_e);
      gather(next_state_layout_, *ctx.nxt, ic, next_e);
      prm_e = gatherParam(ic, *ctx.prm);
      gather(res_layout_, adj, ic, adj_e);
      kernel_.jacobian(ctx.step,
                       ic,
                       wrt,
                       TimeHistoryView(hst_e.data(),
                                       numHistoryStates(),
                                       next_state_layout_.numDofsPerElem()),
                       next_e,
                       prm_e,
                       jac_e);
      matTVec(jac_e, adj_e, col_e);
      assembler.addVec(ic, col_e, out);
    }
  }

#if defined(FEMX_HAS_PETSC)
  if (cell_begin_ != 0 || cell_end_ != numCells())
  {
    allreduce(out);
  }
#endif
}

bool TimeFEMResidual::assembleJac(const TimeContext& ctx,
                                  VariableBlock      wrt,
                                  MatrixBuilder&     out) const
{
  checkContext(ctx);
  const DofLayout* col_layout = layoutFor(wrt);
  if (col_layout == nullptr)
  {
    out.resize(dims().nres, numParams());
    out.setZero();
    return true;
  }

  Assembler assembler(res_layout_, *col_layout, AssemblyMode::Atomic);
  assembler.initMat(out);

#pragma omp parallel
  {
    Vector<Real> hst_e;
    Vector<Real> next_e;
    Vector<Real> prm_e;
    DenseMatrix  jac_e;

#pragma omp for
    for (Index ic = cell_begin_; ic < cell_end_; ++ic)
    {
      gatherHistory(ctx, ic, hst_e);
      gather(next_state_layout_, *ctx.nxt, ic, next_e);
      prm_e = gatherParam(ic, *ctx.prm);
      kernel_.jacobian(ctx.step,
                       ic,
                       wrt,
                       TimeHistoryView(hst_e.data(),
                                       numHistoryStates(),
                                       next_state_layout_.numDofsPerElem()),
                       next_e,
                       prm_e,
                       jac_e);
      assembler.addMat(ic, jac_e, out);
    }
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

Index TimeFEMResidual::numHistoryStates() const
{
  return history_state_layouts_.size();
}

const DofLayout* TimeFEMResidual::layoutFor(VariableBlock wrt) const
{
  if (wrt.isHistoryState())
  {
    const Index lag = wrt.historyLag();
    if (lag < 0 || lag >= numHistoryStates())
    {
      throw runtime_error(
          "TimeFEMResidual history lag is out of range");
    }
    return &history_state_layouts_[lag];
  }
  if (wrt.isNextState())
  {
    return &next_state_layout_;
  }
  return param_layout_ ? &*param_layout_ : nullptr;
}

void TimeFEMResidual::checkLayouts() const
{
  if (nt_ < 0)
  {
    throw runtime_error("TimeFEMResidual received negative step count");
  }
  if (history_state_layouts_.empty())
  {
    throw runtime_error(
        "TimeFEMResidual requires at least one history state layout");
  }
  if (res_layout_.numElems() != next_state_layout_.numElems()
      || (param_layout_ && res_layout_.numElems() != param_layout_->numElems()))
  {
    throw runtime_error(
        "TimeFEMResidual layouts have different cell counts");
  }
  for (const DofLayout& lyt : history_state_layouts_)
  {
    if (res_layout_.numElems() != lyt.numElems())
    {
      throw runtime_error(
          "TimeFEMResidual layouts have different cell counts");
    }
    if (lyt.numDofs() != next_state_layout_.numDofs())
    {
      throw runtime_error(
          "TimeFEMResidual history and next state sizes differ");
    }
    if (lyt.numDofsPerElem() != next_state_layout_.numDofsPerElem())
    {
      throw runtime_error(
          "TimeFEMResidual history and next state local sizes differ");
    }
  }
}

void TimeFEMResidual::checkContext(const TimeContext& ctx) const
{
  const TimeDims dm = dims();
  if (ctx.step < 0 || ctx.step >= dm.nt)
  {
    throw runtime_error("TimeFEMResidual step is out of range");
  }
  const TimeHistoryView hist = ctx.historyView();
  if (hist.count() < dm.nhst
      || hist.stateSize() != dm.nst)
  {
    throw runtime_error(
        "TimeFEMResidual history state size mismatch");
  }
  checkVector(ctx.nxt, dm.nst);
  checkVector(ctx.prm, dm.nprm);
}

void TimeFEMResidual::checkVector(const Vector<Real>* value,
                                  Index               size) const
{
  if (value == nullptr || value->size() != size)
  {
    throw runtime_error("TimeFEMResidual vector size mismatch");
  }
}

void TimeFEMResidual::gatherHistory(const TimeContext& ctx,
                                    Index              ic,
                                    Vector<Real>&      local) const
{
  const Index nloc = next_state_layout_.numDofsPerElem();
  resizeOrZero(local, numHistoryStates() * nloc);
  const TimeHistoryView hist = ctx.historyView();
  for (Index lag = 0; lag < numHistoryStates(); ++lag)
  {
    Vector<Real> state = Vector<Real>::view(local.data() + lag * nloc, nloc);
    gather(history_state_layouts_[lag],
           hist.state(lag),
           ic,
           state);
  }
}

void TimeFEMResidual::checkDirection(VariableBlock       wrt,
                                     const Vector<Real>& dir) const
{
  const DofLayout* lyt = layoutFor(wrt);
  const Index      exp = lyt == nullptr ? numParams() : lyt->numDofs();
  if (dir.size() != exp)
  {
    throw runtime_error("TimeFEMResidual direction size mismatch");
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

void TimeFEMResidual::gather(const DofLayout&    lyt,
                             const Vector<Real>& global,
                             Index               ic,
                             Vector<Real>&       local)
{
  gather(lyt,
         VectorView<const Real>(global.data(), global.size()),
         ic,
         local);
}

void TimeFEMResidual::gather(const DofLayout&       lyt,
                             VectorView<const Real> global,
                             Index                  ic,
                             Vector<Real>&          local)
{
  Vector<Index> dofs;
  lyt.elemDofs(ic, dofs);
  resizeOrZero(local, dofs.size());

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
    throw runtime_error("TimeFEMResidual local matrix size mismatch");
  }
  resizeOrZero(out, mat.rows());
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
    throw runtime_error("TimeFEMResidual local matrix size mismatch");
  }
  resizeOrZero(out, mat.cols());
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

void TimeFEMResidual::checkDof(Index dof, Index size)
{
  if (dof < 0 || dof >= size)
  {
    throw runtime_error("TimeFEMResidual dof is out of range");
  }
}

} // namespace assembly
} // namespace femx
