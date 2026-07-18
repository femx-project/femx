#include <stdexcept>
#include <utility>

#include <femx/assembly/HostTimeResidual.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/MatrixOperator.hpp>
#include <femx/linalg/VectorView.hpp>

#if defined(FEMX_HAS_PETSC)
#include <petscsys.h>
#endif
using namespace femx::state;
using namespace femx::linalg;

namespace femx
{
namespace assembly
{

namespace
{

#if defined(FEMX_HAS_PETSC)
void allreduce(HostVector& out)
{
  const int ierr = MPI_Allreduce(MPI_IN_PLACE,
                                 out.data(),
                                 static_cast<int>(out.size()),
                                 MPIU_REAL,
                                 MPI_SUM,
                                 PETSC_COMM_WORLD);
  if (ierr != MPI_SUCCESS)
  {
    throw std::runtime_error("HostTimeResidual MPI_Allreduce failed");
  }
}
#endif

Array<fem::DofLayout> singleHistLyt(fem::DofLayout lyt)
{
  Array<fem::DofLayout> out;
  out.push_back(std::move(lyt));
  return out;
}

void scatterVec(const fem::DofLayout& lyt,
                Index                 ie,
                const HostVector&     vec_e,
                HostVector&           vec)
{
  Array<Index> dofs;
  lyt.elemDofs(ie, dofs);

  Real* vals = vec.data();
  for (Index i = 0; i < vec_e.size(); ++i)
  {
    const Index id = dofs[i];
#pragma omp atomic update
    vals[id] += vec_e[i];
  }
}

void scatterMat(const fem::DofLayout& row_lyt,
                const fem::DofLayout& col_lyt,
                Index                 ie,
                const DenseMatrix&    mat_e,
                MatrixOperator&       mat)
{
  Array<Index> row_dofs;
  Array<Index> col_dofs;
  row_lyt.elemDofs(ie, row_dofs);
  col_lyt.elemDofs(ie, col_dofs);
  mat.addElem(ie, row_dofs, col_dofs, mat_e, true);
}

} // namespace

HostTimeResidual::HostTimeResidual(Index                      nstep,
                                   fem::DofLayout             res_lyt,
                                   fem::DofLayout             state_lyt,
                                   const HostElementOperator& op)
  : HostTimeResidual(nstep,
                     std::move(res_lyt),
                     singleHistLyt(state_lyt),
                     std::move(state_lyt),
                     op)
{
}

HostTimeResidual::HostTimeResidual(Index                      nstep,
                                   fem::DofLayout             res_lyt,
                                   fem::DofLayout             hist_lyt,
                                   fem::DofLayout             next_lyt,
                                   const HostElementOperator& op)
  : HostTimeResidual(nstep,
                     std::move(res_lyt),
                     singleHistLyt(std::move(hist_lyt)),
                     std::move(next_lyt),
                     op)
{
}

HostTimeResidual::HostTimeResidual(
    Index                      nstep,
    fem::DofLayout             res_lyt,
    Array<fem::DofLayout>      hist_lyts,
    fem::DofLayout             next_lyt,
    const HostElementOperator& op)
  : nstep_(nstep),
    res_lyt_(std::move(res_lyt)),
    hist_lyts_(std::move(hist_lyts)),
    next_lyt_(std::move(next_lyt)),
    op_(op),
    ie_end_(res_lyt_.numElems())
{
  checkLyts();
}

HostTimeResidual::HostTimeResidual(Index                      nstep,
                                   fem::DofLayout             res_lyt,
                                   fem::DofLayout             hist_lyt,
                                   fem::DofLayout             next_lyt,
                                   fem::DofLayout             prm_lyt,
                                   const HostElementOperator& op)
  : HostTimeResidual(nstep,
                     std::move(res_lyt),
                     singleHistLyt(std::move(hist_lyt)),
                     std::move(next_lyt),
                     std::move(prm_lyt),
                     op)
{
}

HostTimeResidual::HostTimeResidual(
    Index                      nstep,
    fem::DofLayout             res_lyt,
    Array<fem::DofLayout>      hist_lyts,
    fem::DofLayout             next_lyt,
    fem::DofLayout             prm_lyt,
    const HostElementOperator& op)
  : nstep_(nstep),
    res_lyt_(std::move(res_lyt)),
    hist_lyts_(std::move(hist_lyts)),
    next_lyt_(std::move(next_lyt)),
    prm_lyt_(std::move(prm_lyt)),
    op_(op),
    ie_end_(res_lyt_.numElems())
{
  checkLyts();
}

void HostTimeResidual::setElemRange(Index ie_begin, Index ie_end)
{
  if (ie_begin < 0 || ie_end < ie_begin || ie_end > numElems())
  {
    throw std::runtime_error("HostTimeResidual received invalid elem range");
  }
#if !defined(FEMX_HAS_PETSC)
  if (ie_begin != 0 || ie_end != numElems())
  {
    throw std::runtime_error("HostTimeResidual elem ranges require PETSc");
  }
#endif
  ie_begin_ = ie_begin;
  ie_end_   = ie_end;
}

TimeDims HostTimeResidual::dims() const
{
  return {nstep_,
          next_lyt_.numDofs(),
          numPrm(),
          res_lyt_.numDofs(),
          numHistStates()};
}

void HostTimeResidual::res(const TimeContext& ctx,
                           HostVector&        out) const
{
  checkCtx(ctx);

  resizeOrZero(out, res_lyt_.numDofs());

#pragma omp parallel
  {
    HostVector hist_e;
    HostVector next_e;
    HostVector prm_e;
    HostVector res_e;

#pragma omp for
    for (Index ie = ie_begin_; ie < ie_end_; ++ie)
    {
      gatherElem(ctx, ie, hist_e, next_e, prm_e);
      op_.res(ctx.step, ie, histView(hist_e), next_e, prm_e, res_e);
      scatterVec(res_lyt_, ie, res_e, out);
    }
  }
  reduce(out);
}

void HostTimeResidual::applyJac(const TimeContext& ctx,
                                VariableBlock      wrt,
                                const HostVector&  dir,
                                HostVector&        out) const
{
  checkCtx(ctx);
  checkDirection(wrt, dir);
  resizeOrZero(out, dims().num_res);

  const fem::DofLayout* col_lyt = layoutFor(wrt);
  if (col_lyt == nullptr)
  {
    return;
  }

#pragma omp parallel
  {
    HostVector  hist_e;
    HostVector  next_e;
    HostVector  prm_e;
    HostVector  dir_e;
    HostVector  res_e;
    DenseMatrix J_e;

#pragma omp for
    for (Index ie = ie_begin_; ie < ie_end_; ++ie)
    {
      gatherElem(ctx, ie, hist_e, next_e, prm_e);
      gather(*col_lyt, dir, ie, dir_e);
      op_.jac(ctx.step, ie, wrt, histView(hist_e), next_e, prm_e, J_e);
      J_e.apply(dir_e, res_e);
      scatterVec(res_lyt_, ie, res_e, out);
    }
  }
  reduce(out);
}

void HostTimeResidual::applyJacT(const TimeContext& ctx,
                                 VariableBlock      wrt,
                                 const HostVector&  adj,
                                 HostVector&        out) const
{
  checkCtx(ctx);
  if (adj.size() != dims().num_res)
  {
    throw std::runtime_error("HostTimeResidual adjoint size mismatch");
  }

  const fem::DofLayout* col_lyt = layoutFor(wrt);
  if (col_lyt == nullptr)
  {
    resizeOrZero(out, numPrm());
    return;
  }

  resizeOrZero(out, col_lyt->numDofs());

#pragma omp parallel
  {
    HostVector  hist_e;
    HostVector  next_e;
    HostVector  prm_e;
    HostVector  adj_e;
    HostVector  col_e;
    DenseMatrix J_e;

#pragma omp for
    for (Index ie = ie_begin_; ie < ie_end_; ++ie)
    {
      gatherElem(ctx, ie, hist_e, next_e, prm_e);
      gather(res_lyt_, adj, ie, adj_e);
      op_.jac(ctx.step, ie, wrt, histView(hist_e), next_e, prm_e, J_e);
      J_e.applyT(adj_e, col_e);
      scatterVec(*col_lyt, ie, col_e, out);
    }
  }
  reduce(out);
}

bool HostTimeResidual::assembleJac(const TimeContext& ctx,
                                   VariableBlock      wrt,
                                   MatrixOperator&    out) const
{
  checkCtx(ctx);
  const fem::DofLayout* col_lyt = layoutFor(wrt);
  if (col_lyt == nullptr)
  {
    out.resize(dims().num_res, numPrm());
    out.setZero();
    return true;
  }

  out.resize(res_lyt_.numDofs(), col_lyt->numDofs());
  out.setZero();

#pragma omp parallel
  {
    HostVector  hist_e;
    HostVector  next_e;
    HostVector  prm_e;
    DenseMatrix J_e;

#pragma omp for
    for (Index ie = ie_begin_; ie < ie_end_; ++ie)
    {
      gatherElem(ctx, ie, hist_e, next_e, prm_e);
      op_.jac(ctx.step, ie, wrt, histView(hist_e), next_e, prm_e, J_e);
      scatterMat(res_lyt_, *col_lyt, ie, J_e, out);
    }
  }
  return true;
}

Index HostTimeResidual::numElems() const
{
  return res_lyt_.numElems();
}

Index HostTimeResidual::numPrm() const
{
  return prm_lyt_ ? prm_lyt_->numDofs() : 0;
}

Index HostTimeResidual::numHistStates() const
{
  return hist_lyts_.size();
}

const fem::DofLayout* HostTimeResidual::layoutFor(VariableBlock wrt) const
{
  if (wrt.isHistoryState())
  {
    const Index lag = wrt.historyLag();
    if (lag < 0 || lag >= numHistStates())
    {
      throw std::runtime_error(
          "HostTimeResidual history lag is out of range");
    }
    return &hist_lyts_[lag];
  }
  if (wrt.isNextState())
  {
    return &next_lyt_;
  }
  return prm_lyt_ ? &*prm_lyt_ : nullptr;
}

void HostTimeResidual::checkLyts() const
{
  if (nstep_ < 0)
  {
    throw std::runtime_error("HostTimeResidual received negative step count");
  }
  if (hist_lyts_.empty())
  {
    throw std::runtime_error(
        "HostTimeResidual requires at least one history state layout");
  }
  if (res_lyt_.numElems() != next_lyt_.numElems()
      || (prm_lyt_ && res_lyt_.numElems() != prm_lyt_->numElems()))
  {
    throw std::runtime_error(
        "HostTimeResidual layouts have different elem counts");
  }
  for (const fem::DofLayout& lyt : hist_lyts_)
  {
    if (res_lyt_.numElems() != lyt.numElems())
    {
      throw std::runtime_error(
          "HostTimeResidual layouts have different elem counts");
    }
    if (lyt.numDofs() != next_lyt_.numDofs())
    {
      throw std::runtime_error(
          "HostTimeResidual history and next state sizes differ");
    }
    if (lyt.numDofsPerElem() != next_lyt_.numDofsPerElem())
    {
      throw std::runtime_error(
          "HostTimeResidual history and next state local sizes differ");
    }
  }
}

void HostTimeResidual::checkCtx(const TimeContext& ctx) const
{
  const TimeDims dm = dims();
  if (ctx.step < 0 || ctx.step >= dm.num_steps)
  {
    throw std::runtime_error("HostTimeResidual step is out of range");
  }
  const TimeHistoryView hist = ctx.hist;
  if (hist.count() < dm.num_history_states
      || hist.stateSize() != dm.num_states)
  {
    throw std::runtime_error(
        "HostTimeResidual history state size mismatch");
  }
  if (ctx.nxt == nullptr || ctx.nxt->size() != dm.num_states
      || ctx.prm == nullptr || ctx.prm->size() != dm.num_param)
  {
    throw std::runtime_error("HostTimeResidual vector size mismatch");
  }
}

void HostTimeResidual::gatherHist(const TimeContext& ctx,
                                  Index              ie,
                                  HostVector&        hist_e) const
{
  const Index ndof_e = next_lyt_.numDofsPerElem();
  resizeOrZero(hist_e, numHistStates() * ndof_e);
  const TimeHistoryView hist = ctx.hist;
  for (Index lag = 0; lag < numHistStates(); ++lag)
  {
    HostVectorView state_e(hist_e.data() + lag * ndof_e, ndof_e);
    gather(hist_lyts_[lag], hist.state(lag), ie, state_e);
  }
}

void HostTimeResidual::gatherElem(const TimeContext& ctx,
                                  Index              ie,
                                  HostVector&        hist_e,
                                  HostVector&        next_e,
                                  HostVector&        prm_e) const
{
  gatherHist(ctx, ie, hist_e);
  gather(next_lyt_, *ctx.nxt, ie, next_e);
  if (prm_lyt_)
  {
    gather(*prm_lyt_, *ctx.prm, ie, prm_e);
  }
  else
  {
    prm_e.resize(0);
  }
}

TimeHistoryView HostTimeResidual::histView(const HostVector& hist_e) const
{
  return TimeHistoryView(hist_e.data(),
                         numHistStates(),
                         next_lyt_.numDofsPerElem());
}

void HostTimeResidual::checkDirection(VariableBlock     wrt,
                                      const HostVector& dir) const
{
  const fem::DofLayout* lyt = layoutFor(wrt);
  const Index           exp = lyt == nullptr ? numPrm() : lyt->numDofs();
  if (dir.size() != exp)
  {
    throw std::runtime_error("HostTimeResidual direction size mismatch");
  }
}

void HostTimeResidual::reduce(HostVector& vec) const
{
#if defined(FEMX_HAS_PETSC)
  if (ie_begin_ != 0 || ie_end_ != numElems())
  {
    allreduce(vec);
  }
#else
  (void) vec;
#endif
}

void HostTimeResidual::gather(const fem::DofLayout& lyt,
                              const HostVector&     vec,
                              Index                 ie,
                              HostVector&           vec_e)
{
  gather(lyt, HostConstVectorView(vec.data(), vec.size()), ie, vec_e);
}

void HostTimeResidual::gather(const fem::DofLayout& lyt,
                              HostConstVectorView   vec,
                              Index                 ie,
                              HostVector&           vec_e)
{
  Array<Index> dofs;
  lyt.elemDofs(ie, dofs);
  resizeOrZero(vec_e, dofs.size());

  gather(lyt, vec, ie, HostVectorView(vec_e.data(), vec_e.size()));
}

void HostTimeResidual::gather(const fem::DofLayout& lyt,
                              HostConstVectorView   vec,
                              Index                 ie,
                              HostVectorView        vec_e)
{
  Array<Index> dofs;
  lyt.elemDofs(ie, dofs);
  for (Index i = 0; i < vec_e.size(); ++i)
  {
    vec_e[i] = vec[dofs[i]];
  }
}

} // namespace assembly
} // namespace femx
