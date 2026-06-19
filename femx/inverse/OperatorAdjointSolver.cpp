#include <stdexcept>

#include <femx/eq/StateJacobianOperator.hpp>
#include <femx/inverse/OperatorAdjointSolver.hpp>

using namespace femx::eq;
using namespace femx::system;

namespace femx
{
namespace inverse
{

OperatorAdjointSolver::OperatorAdjointSolver(const ResidualEquation& eq,
                                             LinearSolver&       lin_solver)
  : eq_(eq),
    lin_solver_(lin_solver)
{
  if (eq_.numRes() != eq_.numStates())
  {
    throw std::runtime_error(
        "OperatorAdjointSolver requires square state residual dimensions");
  }
}

Index OperatorAdjointSolver::numStates() const
{
  return eq_.numStates();
}

Index OperatorAdjointSolver::numParams() const
{
  return eq_.numParams();
}

Index OperatorAdjointSolver::numRes() const
{
  return eq_.numRes();
}

void OperatorAdjointSolver::solve(const Vector<Real>& state,
                                  const Vector<Real>& prm,
                                  const Vector<Real>& rhs,
                                  Vector<Real>&       adjoint)
{
  if (state.size() != numStates() || prm.size() != numParams()
      || rhs.size() != numStates())
  {
    throw std::runtime_error("OperatorAdjointSolver size mismatch");
  }

  const StateJacobianOperator jac(eq_, state, prm);
  lin_solver_.solveT(jac, rhs, adjoint);
  if (adjoint.size() != numRes())
  {
    throw std::runtime_error("OperatorAdjointSolver adjoint size mismatch");
  }
}

} // namespace inverse
} // namespace femx
