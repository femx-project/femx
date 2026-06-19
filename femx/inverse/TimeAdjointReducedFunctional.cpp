#include <stdexcept>

#include <femx/eq/TimeStateJacobianOperator.hpp>
#include <femx/inverse/TimeAdjointReducedFunctional.hpp>

using namespace femx::eq;
using namespace femx::system;

namespace femx
{
namespace inverse
{

TimeAdjointReducedFunctional::TimeAdjointReducedFunctional(
    TimeStateSolver&            state_solver,
    const TimeResidualEquation& eq,
    LinearSolver&           adjoint_solver,
    const TimeObjectiveFunctional&  obj)
  : state_solver_(state_solver),
    eq_(eq),
    adj_solver_(adjoint_solver),
    obj_(obj)
{
  checkDims();
}

Index TimeAdjointReducedFunctional::numParams() const
{
  return state_solver_.numParams();
}

Real TimeAdjointReducedFunctional::value(const Vector<Real>& prm)
{
  TimeStateTrajectory tr;
  solveFwd(prm, tr);
  return obj_.value(tr, prm);
}

void TimeAdjointReducedFunctional::grad(const Vector<Real>& prm,
                                        Vector<Real>&       out)
{
  TimeStateTrajectory tr;
  solveFwd(prm, tr);
  gradAt(tr, prm, out);
}

Real TimeAdjointReducedFunctional::valueGrad(const Vector<Real>& prm,
                                             Vector<Real>&       grad_out)
{
  TimeStateTrajectory tr;
  solveFwd(prm, tr);
  const Real obj_val = obj_.value(tr, prm);
  gradAt(tr, prm, grad_out);
  return obj_val;
}

void TimeAdjointReducedFunctional::checkDims() const
{
  if (state_solver_.numSteps() != eq_.numSteps()
      || state_solver_.numSteps() != obj_.numSteps()
      || state_solver_.numStates() != eq_.numStates()
      || state_solver_.numStates() != obj_.numStates()
      || state_solver_.numParams() != eq_.numParams()
      || state_solver_.numParams() != obj_.numParams()
      || eq_.numRes() != eq_.numStates())
  {
    throw std::runtime_error(
        "TimeAdjointReducedFunctional received inconsistent dimensions");
  }
}

void TimeAdjointReducedFunctional::solveFwd(
    const Vector<Real>&      prm,
    TimeStateTrajectory& tr)
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "TimeAdjointReducedFunctional parameter size mismatch");
  }

  state_solver_.solve(prm, tr);
  if (tr.numSteps() != state_solver_.numSteps()
      || tr.numStates() != state_solver_.numStates())
  {
    throw std::runtime_error(
        "TimeAdjointReducedFunctional forward trajectory size mismatch");
  }
}

void TimeAdjointReducedFunctional::gradAt(
    const TimeStateTrajectory& tr,
    const Vector<Real>&            prm,
    Vector<Real>&                  out)
{
  obj_.paramGrad(tr, prm, out);
  if (out.size() != numParams())
  {
    throw std::runtime_error(
        "TimeAdjointReducedFunctional parameter gradient size mismatch");
  }

  const Index  num_steps = state_solver_.numSteps();
  Vector<Real> next_adj;

  Vector<Real> rhs;
  Vector<Real> carry;
  Vector<Real> adj;
  Vector<Real> param_adj;

  for (Index step = num_steps; step-- > 0;)
  {
    obj_.stateGrad(step + 1, tr, prm, rhs);
    checkSize(rhs,
              eq_.numStates(),
              "TimeAdjointReducedFunctional state gradient size mismatch");

    if (step + 1 < num_steps)
    {
      eq_.applyPreviousStateJacT(
          step + 1,
          tr[step + 2],
          tr[step + 1],
          prm,
          next_adj,
          carry);
      checkSize(carry,
                eq_.numStates(),
                "TimeAdjointReducedFunctional carry size mismatch");
      for (Index i = 0; i < rhs.size(); ++i)
      {
        rhs[i] -= carry[i];
      }
    }

    const TimeStateJacobianOperator next_jac(
        eq_, step, tr[step + 1], tr[step], prm, TimeStateSlot::Next);

    adj_solver_.solveT(next_jac, rhs, adj);
    checkSize(adj,
              eq_.numRes(),
              "TimeAdjointReducedFunctional adjoint size mismatch");

    eq_.applyParamJacT(
        step, tr[step + 1], tr[step], prm, adj, param_adj);
    checkSize(param_adj,
              numParams(),
              "TimeAdjointReducedFunctional residual parameter gradient size mismatch");
    for (Index i = 0; i < out.size(); ++i)
    {
      out[i] -= param_adj[i];
    }
    next_adj = adj;
  }
}

void TimeAdjointReducedFunctional::checkSize(const Vector<Real>& value,
                                             Index               expected,
                                             const char*         message)
{
  if (value.size() != expected)
  {
    throw std::runtime_error(message);
  }
}

} // namespace inverse
} // namespace femx
