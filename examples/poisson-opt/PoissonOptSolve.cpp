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
  ctx.vectorHandler().copy(h_prm.view(), prm);
  state_solver.solve(state, prm);

  HostVector<Real> h_state;
  ctx.vectorHandler().copy(state.view(), h_state);
  ctx.sync();
  return h_state;
}

template <MemorySpace Space>
Result optimizeImpl(
    PoissonOptProblem&           problem,
    state::StateSolver<Space>&   state_solver,
    linalg::LinearSystem<Space>& adj_system,
    MPI_Comm                     comm)
{
  problem.prepareObjective(
      solveState(state_solver, problem.targetControl()));

  inverse::ReducedFunctional<Space> functional(
      state_solver, adj_system, problem.objective());

  opt::TaoOptimizer optimizer(functional, comm);
  optimizer.opts().max_its = problem.options().max_iterations;

  const HostVector<Real> init_ctr(problem.numParameters(), 0.0);
  opt::TaoResult         tao_res;

  runtime::checkPetsc(
      optimizer.solve(init_ctr, tao_res),
      "TaoOptimizer::solve");

  HostVector<Real> final_state = solveState(state_solver, tao_res.prm);

  Result result;

  result.report     = problem.report(tao_res.prm, final_state, tao_res.value, tao_res.grad);
  result.control    = std::move(tao_res.prm);
  result.state      = std::move(final_state);
  result.gradient   = std::move(tao_res.grad);
  result.iterations = tao_res.its;
  result.reason     = static_cast<int>(tao_res.reason);
  result.converged  = tao_res.converged();

  return result;
}

} // namespace

Result optimize(
    PoissonOptProblem&                       problem,
    state::StateSolver<MemorySpace::Host>&   state_solver,
    linalg::LinearSystem<MemorySpace::Host>& adj_system,
    MPI_Comm                                 comm)
{
  return optimizeImpl(
      problem, state_solver, adj_system, comm);
}

#if defined(FEMX_HAS_CUDA)

Result optimize(
    PoissonOptProblem&                         problem,
    state::StateSolver<MemorySpace::Device>&   state_solver,
    linalg::LinearSystem<MemorySpace::Device>& adj_system,
    MPI_Comm                                   comm)
{
  return optimizeImpl(
      problem, state_solver, adj_system, comm);
}

#endif

} // namespace femx::examples::poisson_opt
