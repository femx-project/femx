#include <stdexcept>

#include <femx/solve/MatrixAdjointSolver.hpp>

using namespace femx::solve;
using namespace femx::problem;
using namespace femx::algebra;

namespace femx
{
namespace solve
{

MatrixAdjointSolver::MatrixAdjointSolver(
    const MatrixResidualEquation& eq,
    SystemMatrix&                 state_jac,
    LinearSolver&                 lin_solver)
  : eq_(eq),
    state_jac_(state_jac),
    lin_solver_(lin_solver)
{
  if (eq_.numRes() != eq_.numStates())
  {
    throw std::runtime_error(
        "MatrixAdjointSolver requires square state residual dimensions");
  }
}

Index MatrixAdjointSolver::numStates() const
{
  return eq_.numStates();
}

Index MatrixAdjointSolver::numParams() const
{
  return eq_.numParams();
}

Index MatrixAdjointSolver::numRes() const
{
  return eq_.numRes();
}

void MatrixAdjointSolver::solve(const Vector<Real>& state,
                                const Vector<Real>& prm,
                                const Vector<Real>& rhs,
                                Vector<Real>&       adjoint)
{
  if (state.size() != numStates() || prm.size() != numParams()
      || rhs.size() != numStates())
  {
    throw std::runtime_error("MatrixAdjointSolver size mismatch");
  }

  eq_.assembleStateJac(state, prm, state_jac_);
  state_jac_.finalize();
  lin_solver_.solveT(state_jac_, rhs, adjoint);
  if (adjoint.size() != numRes())
  {
    throw std::runtime_error("MatrixAdjointSolver adjoint size mismatch");
  }
}

} // namespace solve
} // namespace femx
