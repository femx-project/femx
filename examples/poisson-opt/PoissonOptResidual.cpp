#include "PoissonOptResidual.hpp"

#include "../poisson/ElementKernel.hpp"
#include <femx/ad/Enzyme.hpp>
#include <femx/assembly/Assembly.hpp>
#include <femx/common/Checks.hpp>

namespace femx::examples::poisson_opt
{
namespace
{

Real controlDerivative(Real state, Real control)
{
#if defined(FEMX_HAS_ENZYME)
  return __enzyme_fwddiff<Real>(
      reinterpret_cast<void*>(detail::controlResidual),
      enzyme_const,
      state,
      enzyme_dup,
      control,
      1.0);
#else
  (void) state;
  (void) control;
  return -1.0;
#endif
}

} // namespace

HostPoissonOptResidual::HostPoissonOptResidual(
    const PoissonOptProblem& prob)
  : prob_(prob)
{
}

state::Dimensions HostPoissonOptResidual::dims() const
{
  return {prob_.numStates(),
          prob_.numParameters(),
          prob_.numStates()};
}

const HostCsrPattern& HostPoissonOptResidual::hostPattern() const
{
  return prob_.assemblyMap().pattern();
}

void HostPoissonOptResidual::assembleResidual(
    const HostVector<Real>&             state,
    const HostVector<Real>&             prm,
    HostVector<Real>&                   out,
    linalg::Context<MemorySpace::Host>& ctx) const
{
  checkVectors(state, prm);
  assembly::assembleResidual(
      poisson::ElementKernel<MemorySpace::Host>(
          prob_.elementData().view()),
      prob_.mesh(),
      prob_.assemblyMap(),
      state,
      out,
      ctx);

  const HostVector<Real> vals = boundaryValues(prm);
  assembly::applyDirichletConditions(prob_.boundaryMap(),
                                     state.view(),
                                     vals.view(),
                                     out.view());
}

void HostPoissonOptResidual::assembleJacobian(
    const HostVector<Real>&              state,
    const HostVector<Real>&              prm,
    linalg::Jacobian<MemorySpace::Host>& out,
    linalg::Context<MemorySpace::Host>&  ctx) const
{
  checkVectors(state, prm);
  assembly::assembleJacobian(
      poisson::ElementKernel<MemorySpace::Host>(
          prob_.elementData().view()),
      prob_.mesh(),
      prob_.assemblyMap(),
      state,
      out,
      ctx);
  assembly::applyDirichletConditions(prob_.boundaryMap(), out);
}

void HostPoissonOptResidual::applyParamJacT(
    const HostVector<Real>& state,
    const HostVector<Real>& prm,
    const HostVector<Real>& adj,
    HostVector<Real>&       out,
    linalg::Context<MemorySpace::Host>&) const
{
  checkVectors(state, prm);
  require(adj.size() == dims().num_res,
          "Poisson optimization adjoint size mismatch");

  out.resize(prob_.numParameters());
  for (Index ip = 0; ip < prob_.numParameters(); ++ip)
  {
    const Index state_dof = prob_.controlDofs()[ip];
    out[ip] =
        controlDerivative(state[state_dof], prm[ip]) * adj[state_dof];
  }
}

void HostPoissonOptResidual::checkVectors(
    const HostVector<Real>& state,
    const HostVector<Real>& prm) const
{
  require(state.size() == dims().num_states,
          "Poisson optimization state size mismatch");
  require(prm.size() == dims().num_param,
          "Poisson optimization parameter size mismatch");
}

HostVector<Real> HostPoissonOptResidual::boundaryValues(
    const HostVector<Real>& prm) const
{
  HostVector<Real> vals(prob_.boundaryMap().numBcs(), 0.0);
  for (Index ip = 0; ip < prob_.numParameters(); ++ip)
  {
    vals[ip] = prm[ip];
  }

  return vals;
}

} // namespace femx::examples::poisson_opt
