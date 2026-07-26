#include "PoissonResidual.hpp"

#include "ElementKernel.hpp"
#include <femx/assembly/Assembly.hpp>

namespace femx::examples::poisson
{

HostPoissonResidual::HostPoissonResidual(const PoissonProblem& prob)
  : prob_(prob)
{
}

state::Dimensions HostPoissonResidual::dims() const
{
  return {prob_.numDofs(), 0, prob_.numDofs()};
}

const HostCsrPattern& HostPoissonResidual::hostPattern() const
{
  return prob_.assemblyMap().pattern();
}

void HostPoissonResidual::assembleResidual(
    const HostVector<Real>& state,
    const HostVector<Real>& /* prm */,
    HostVector<Real>&                   out,
    linalg::Context<MemorySpace::Host>& ctx) const
{
  assembly::assembleResidual(
      ElementKernel<MemorySpace::Host>(prob_.elementData().view()),
      prob_.mesh(),
      prob_.assemblyMap(),
      state,
      out,
      ctx);
  assembly::applyDirichletConditions(prob_.boundaryMap(),
                                     state.view(),
                                     prob_.boundaryValues().view(),
                                     out.view());
}

void HostPoissonResidual::assembleJacobian(
    const HostVector<Real>& state,
    const HostVector<Real>& /* prm */,
    linalg::Jacobian<MemorySpace::Host>& out,
    linalg::Context<MemorySpace::Host>&  ctx) const
{
  assembly::assembleJacobian(
      ElementKernel<MemorySpace::Host>(prob_.elementData().view()),
      prob_.mesh(),
      prob_.assemblyMap(),
      state,
      out,
      ctx);
  assembly::applyDirichletConditions(prob_.boundaryMap(), out);
}

void HostPoissonResidual::applyParamJacT(
    const HostVector<Real>& /* state */,
    const HostVector<Real>& /* prm */,
    const HostVector<Real>& /* adj */,
    HostVector<Real>& out,
    linalg::Context<MemorySpace::Host>&) const
{
  // This example has no parameters.
  out.resize(0);
}

} // namespace femx::examples::poisson
