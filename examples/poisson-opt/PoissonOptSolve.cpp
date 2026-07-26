#include "PoissonOptSolve.hpp"

#include <utility>

#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/opt/TaoOptimizer.hpp>
#include <femx/runtime/PETScRuntime.hpp>

namespace femx::examples::poisson_opt
{
namespace
{

template <MemorySpace Space>
HostVector<Real> solveState(
    state::StateSolver<Space>& state_solver,
    const HostVector<Real>&    h_prm)
{
  using Vec = Vector<Space, Real>;

  auto& ctx = state_solver.context();
  Vec   prm;
  Vec   state;
  ctx.vectors().copy(h_prm.view(), prm);
  state_solver.solve(prm, state);

  HostVector<Real> h_state;
  ctx.vectors().copy(state.view(), h_state);
  ctx.sync();
  return h_state;
}

template <MemorySpace Space>
Result optimizeImpl(
    PoissonOptProblem&           prob,
    state::StateSolver<Space>&   state_solver,
    linalg::LinearSystem<Space>& adj_system,
    MPI_Comm                     comm)
{
  prob.prepareObjective(
      solveState(state_solver, prob.targetControl()));

  inverse::ReducedFunctional<Space> functional(
      state_solver, adj_system, prob.objective());
  opt::TaoOptimizer optimizer(functional, comm);
  optimizer.opts().max_its = prob.options().max_iterations;

  const HostVector<Real> init_control(prob.numParameters(), 0.0);
  opt::TaoResult         tao_res;

  runtime::checkPetsc(
      optimizer.solve(init_control, tao_res),
      "TaoOptimizer::solve");

  HostVector<Real> final_state = solveState(state_solver, tao_res.prm);

  Result sol;
  sol.report     = prob.report(tao_res.prm, final_state, tao_res.value, tao_res.grad);
  sol.control    = std::move(tao_res.prm);
  sol.state      = std::move(final_state);
  sol.gradient   = std::move(tao_res.grad);
  sol.iterations = tao_res.its;
  sol.reason     = static_cast<int>(tao_res.reason);
  sol.converged  = tao_res.converged();
  return sol;
}

} // namespace

Result optimize(
    PoissonOptProblem&                       prob,
    state::StateSolver<MemorySpace::Host>&   state_solver,
    linalg::LinearSystem<MemorySpace::Host>& adj_system,
    MPI_Comm                                 comm)
{
  return optimizeImpl(
      prob, state_solver, adj_system, comm);
}

#if defined(FEMX_HAS_CUDA)

Result optimize(
    PoissonOptProblem&                         prob,
    state::StateSolver<MemorySpace::Device>&   state_solver,
    linalg::LinearSystem<MemorySpace::Device>& adj_system,
    MPI_Comm                                   comm)
{
  return optimizeImpl(
      prob, state_solver, adj_system, comm);
}

#endif

} // namespace femx::examples::poisson_opt
