#include "NavierResidual.hpp"

#include <stdexcept>

#include <femx/ad/Enzyme.hpp>
#include <femx/assembly/Assembly.hpp>
#include <femx/common/Checks.hpp>
#include <femx/model/navier/NavierModel.hpp>

namespace femx::model::navier
{
namespace
{

void validateTimeContext(const state::HostTimeContext& time,
                         Index                         num_steps,
                         Index                         num_states)
{
  require(time.step >= 0 && time.step < num_steps,
          "Navier-Stokes residual step is out of range");
  require(time.hist.count() >= kNumHist
              && time.hist.stateSize() == num_states
              && time.nxt.size() == num_states && time.prm.empty(),
          "Navier-Stokes residual vector size mismatch");
}

struct NavierWork
{
  HostVector<Real> hist;
  HostVector<Real> next;
  HostVector<Real> adj;
  HostVector<Real> product;
};

void gatherElement(const assembly::HostAssemblyMap& map,
                   HostVectorView<const Real>       hist,
                   HostVectorView<const Real>       next,
                   Index                            elem,
                   NavierWork&                      work)
{
  const auto  map_view       = map.view();
  const Index num_state_dofs = map_view.numStateDofs(elem);

  work.hist.resize(kNumHist * num_state_dofs);
  work.next.resize(num_state_dofs);

  for (Index lag = 0; lag < kNumHist; ++lag)
  {
    for (Index j = 0; j < num_state_dofs; ++j)
    {
      const Index dof = map_view.stateDof(elem, j);
      work.hist[lag * num_state_dofs + j] =
          hist[lag * map.numStates() + dof];
    }
  }
  for (Index j = 0; j < num_state_dofs; ++j)
  {
    work.next[j] = next[map_view.stateDof(elem, j)];
  }
}

void add(HostVector<Real>& vals, Index idx, Real val)
{
#pragma omp atomic update
  vals[idx] += val;
}

void applyHistoryJacT(
    const HostNavierElementKernel&      kernel,
    const assembly::HostAssemblyMap&    map,
    const state::HostTimeContext&       time,
    Index                               lag,
    HostVectorView<const Real>          adj,
    HostVector<Real>&                   out,
    linalg::Context<MemorySpace::Host>& ctx)
{
  ctx.vectorHandler().assign(out, map.numStates(), 0);

  const auto hist =
      HostVectorView<const Real>(time.hist.data(),
                                 kNumHist * map.numStates());
  const auto range = ctx.elementRange(map.numElems());

#pragma omp parallel
  {
    NavierWork work;
#pragma omp for
    for (Index ie = range.begin; ie < range.end; ++ie)
    {
      const auto                        map_view = map.view();
      const HostVectorView<const Index> element_rows(
          map_view.res_dofs + map_view.res_offsets[ie],
          map_view.numResDofs(ie));
      if (!ctx.ownsElement(ie, map.numElems(), element_rows))
      {
        continue;
      }

      gatherElement(map, hist, time.nxt, ie, work);

      const Index num_res_dofs   = map_view.numResDofs(ie);
      const Index num_state_dofs = map_view.numStateDofs(ie);
      work.adj.resize(num_res_dofs);
      work.product.resize(kNumHist * num_state_dofs);

      for (Index i = 0; i < num_res_dofs; ++i)
      {
        work.adj[i] = adj[map_view.resDof(ie, i)];
      }

      const assembly::HostTimeElementView element_view{
          ie,
          time.step,
          kNumHist,
          work.hist.view(),
          work.next.view()};
      histVjp(
          kernel, element_view, work.adj.view(), work.product.view());

      for (Index j = 0; j < num_state_dofs; ++j)
      {
        add(out,
            map_view.stateDof(ie, j),
            work.product[lag * num_state_dofs + j]);
      }
    }
  }
  ctx.allReduceSum(out.view());
}

} // namespace

HostNavierResidual::HostNavierResidual(
    const NavierModel& model)
  : model_(model)
{
}

state::TimeDims HostNavierResidual::dims() const
{
  return {model_.numSteps(),
          model_.numStates(),
          0,
          model_.assemblyMap().numRes(),
          kNumHist};
}

const HostCsrPattern& HostNavierResidual::hostPattern() const
{
  return model_.assemblyMap().pattern();
}

void HostNavierResidual::initialState(
    ConstView prm,
    Vec&      out,
    Ctx&      ctx) const
{
  require(prm.empty(),
          "Navier-Stokes physics residual is parameter-free");
  ctx.vectorHandler().assign(out, model_.numStates(), 0);
}

void HostNavierResidual::assembleNext(
    const StepCtx& time,
    Vec&           res,
    Jac&           jac,
    Ctx&           ctx) const
{
  validateTimeContext(time, model_.numSteps(), model_.numStates());

  const auto& map   = model_.assemblyMap();
  const auto  range = ctx.elementRange(map.numElems());
  const auto  hist  = ConstView(
      time.hist.data(), kNumHist * map.numStates());

  assembly::assembleResidualAndJacobian(
      HostNavierElementKernel(model_.elementData().view(), model_.fluid(), model_.dt()),
      time.step,
      kNumHist,
      state::VariableBlock::NextState,
      map,
      range.begin,
      range.end,
      hist,
      time.nxt,
      res,
      jac,
      ctx);
}

void HostNavierResidual::applyJacT(
    const StepCtx&       time,
    state::VariableBlock with_respect_to,
    ConstView            adj,
    Vec&                 out,
    Ctx&                 ctx) const
{
  validateTimeContext(time, model_.numSteps(), model_.numStates());

  require(!with_respect_to.isNextState(),
          "Navier-Stokes transpose apply expects a history or parameter block");
  require(adj.size() == model_.assemblyMap().numRes(),
          "Navier-Stokes residual adjoint size mismatch");

  if (with_respect_to.isParam())
  {
    out.resize(0);
    return;
  }

  require(with_respect_to.historyLag() >= 0
              && with_respect_to.historyLag() < kNumHist,
          "Navier-Stokes residual history lag is out of range");

  if (!ad::has_enzyme)
  {
    throw std::runtime_error(
        "Navier-Stokes history VJP requires Enzyme");
  }

  applyHistoryJacT(
      HostNavierElementKernel(model_.elementData().view(), model_.fluid(), model_.dt()),
      model_.assemblyMap(),
      time,
      with_respect_to.historyLag(),
      adj,
      out,
      ctx);
}

} // namespace femx::model::navier
