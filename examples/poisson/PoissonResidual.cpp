#include "PoissonResidual.hpp"

#include "PoissonElementKernel.hpp"
#include <femx/assembly/Assembly.hpp>

namespace femx::examples::poisson
{

HostPoissonResidual::HostPoissonResidual(const PoissonProblem& problem)
  : problem_(problem)
{
}

state::Dimensions HostPoissonResidual::dims() const
{
  return {problem_.numDofs(), 0, problem_.numDofs()};
}

const HostCsrPattern& HostPoissonResidual::hostPattern() const
{
  return problem_.assemblyMap().pattern();
}

void HostPoissonResidual::assembleResidual(
    const HostVector<Real>& state,
    const HostVector<Real>& /* prm */,
    HostVector<Real>&                   out,
    linalg::Context<MemorySpace::Host>& ctx) const
{
  // Assemble A x from the element residuals R_i^e = sum_j A_ij^e x_j^e.
  assembly::assembleResidual(
      HostPoissonElementKernel(problem_.elementData().view()),
      problem_.mesh(),
      problem_.assemblyMap(),
      state,
      out,
      ctx);

  // Apply R_i(x) = x_i - g_i at prescribed rows to obtain the residual
  // R(x) = A x - b of the Dirichlet-constrained system.
  assembly::applyDirichletConditions(problem_.boundaryMap(),
                                     state.view(),
                                     problem_.boundaryValues().view(),
                                     out.view());
}

void HostPoissonResidual::assembleJacobian(
    const HostVector<Real>& state,
    const HostVector<Real>& /* prm */,
    linalg::SystemMatrix<MemorySpace::Host>& out,
    linalg::Context<MemorySpace::Host>&      ctx) const
{
  // Assemble dR/dx = A from the element matrices A^e.
  assembly::assembleJacobian(
      HostPoissonElementKernel(problem_.elementData().view()),
      problem_.mesh(),
      problem_.assemblyMap(),
      state,
      out,
      ctx);

  // A constrained residual is x_i - g_i, hence its Jacobian row is the
  // corresponding identity row.
  assembly::applyDirichletConditions(problem_.boundaryMap(), out);
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
