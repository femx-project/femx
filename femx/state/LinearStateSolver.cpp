#include <stdexcept>

#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/LinearStateSolver.hpp>
#include <femx/state/Linearization.hpp>
using namespace femx::state;
using namespace femx::linalg;

namespace femx
{
namespace state
{

LinearStateSolver::LinearStateSolver(const Residual& problem,
                                     Linearization&  lin,
                                     LinearSolver&   lin_solver)
  : problem_(problem),
    linearization_(lin),
    lin_solver_(lin_solver),
    dims_(problem.dims())
{
  if (dims_.num_res != dims_.num_states)
  {
    throw std::runtime_error(
        "LinearStateSolver requires square state residual dimensions");
  }
}

Index LinearStateSolver::numStates() const
{
  return dims_.num_states;
}

Index LinearStateSolver::numParams() const
{
  return dims_.num_param;
}

Index LinearStateSolver::numRes() const
{
  return dims_.num_res;
}

void LinearStateSolver::solve(const HostVector& prm,
                              HostVector&       state)
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error("LinearStateSolver parameter size mismatch");
  }

  HostVector zero_state(numStates());
  HostVector res;
  problem_.res(zero_state, prm, res);
  if (res.size() != numRes())
  {
    throw std::runtime_error("LinearStateSolver residual size mismatch");
  }

  HostVector rhs(res.size());
  for (Index i = 0; i < res.size(); ++i)
  {
    rhs[i] = -res[i];
  }

  problem_.linearize(zero_state, prm, linearization_);
  lin_solver_.solve(linearization_.stateJac(), rhs, state);
  if (state.size() != numStates())
  {
    throw std::runtime_error("LinearStateSolver solution size mismatch");
  }
}

} // namespace state
} // namespace femx
