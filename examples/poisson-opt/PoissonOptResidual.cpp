#include "PoissonOptResidual.hpp"

#include "../poisson/PoissonElementKernel.hpp"
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
    const PoissonOptProblem& problem)
  : problem_(problem)
{
}

state::Dimensions HostPoissonOptResidual::dims() const
{
  return {problem_.numStates(),
          problem_.numParameters(),
          problem_.numStates()};
}

const HostCsrPattern& HostPoissonOptResidual::hostPattern() const
{
  return problem_.assemblyMap().pattern();
}

void HostPoissonOptResidual::assembleResidual(
    const HostVector<Real>&             state,
    const HostVector<Real>&             prm,
    HostVector<Real>&                   out,
    linalg::Context<MemorySpace::Host>& ctx) const
{
  checkVectors(state, prm);

  assembly::assembleResidual(
      poisson::HostPoissonElementKernel(
          problem_.elementData().view()),
      problem_.mesh(),
      problem_.assemblyMap(),
      state,
      out,
      ctx);

  const HostVector<Real> vals = boundaryValues(prm);
  assembly::applyDirichletConditions(problem_.boundaryMap(),
                                     state.view(),
                                     vals.view(),
                                     out.view());
}

void HostPoissonOptResidual::assembleJacobian(
    const HostVector<Real>&                  state,
    const HostVector<Real>&                  prm,
    linalg::SystemMatrix<MemorySpace::Host>& out,
    linalg::Context<MemorySpace::Host>&      ctx) const
{
  checkVectors(state, prm);
  assembly::assembleJacobian(
      poisson::HostPoissonElementKernel(
          problem_.elementData().view()),
      problem_.mesh(),
      problem_.assemblyMap(),
      state,
      out,
      ctx);
  assembly::applyDirichletConditions(problem_.boundaryMap(), out);
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

  out.resize(problem_.numParameters());
  for (Index ip = 0; ip < problem_.numParameters(); ++ip)
  {
    const Index state_dof = problem_.controlDofs()[ip];
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
  HostVector<Real> vals(problem_.boundaryMap().numBcs(), 0.0);
  for (Index ip = 0; ip < problem_.numParameters(); ++ip)
  {
    vals[ip] = prm[ip];
  }

  return vals;
}

} // namespace femx::examples::poisson_opt
