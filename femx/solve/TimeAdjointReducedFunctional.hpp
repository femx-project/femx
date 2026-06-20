#pragma once

#include <vector>

#include <femx/core/Types.hpp>
#include <femx/problem/TimeResidualEquation.hpp>
#include <femx/solve/TimeStateSolver.hpp>
#include <femx/solve/TimeStateTrajectory.hpp>
#include <femx/solve/ReducedObjective.hpp>
#include <femx/problem/TimeObjectiveFunctional.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/LinearSolver.hpp>

namespace femx
{
namespace solve
{

using problem::TimeObjectiveFunctional;

/**
 * @brief Reduced objective for time-marching equations using a discrete adjoint.
 *
 * The residual convention is R_k(u_{k+1}, u_k, m) = 0 for
 * k = 0, ..., N - 1. The initial state u_0 is assumed fixed with respect to
 * the reduced parameters.
 */
class TimeAdjointReducedFunctional final : public ReducedObjective
{
public:
  TimeAdjointReducedFunctional(solve::TimeStateSolver&            state_solver,
                               const problem::TimeResidualEquation& eq,
                               algebra::LinearSolver&           adjoint_solver,
                               const TimeObjectiveFunctional&  obj);

  Index numParams() const override;

  Real value(const Vector<Real>& prm) override;

  void grad(const Vector<Real>& prm,
            Vector<Real>&       out) override;

  Real valueGrad(const Vector<Real>& prm,
                 Vector<Real>&       grad_out) override;

private:
  void checkDims() const;

  void solveFwd(const Vector<Real>&      prm,
                solve::TimeStateTrajectory& tr);

  void gradAt(const solve::TimeStateTrajectory& tr,
              const Vector<Real>&            prm,
              Vector<Real>&                  out);

  static void checkSize(const Vector<Real>& value,
                        Index               expected,
                        const char*         message);

private:
  solve::TimeStateSolver&            state_solver_;
  const problem::TimeResidualEquation& eq_;
  algebra::LinearSolver&           adj_solver_;
  const TimeObjectiveFunctional&  obj_;
};

} // namespace solve
} // namespace femx
