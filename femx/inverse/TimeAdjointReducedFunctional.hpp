#pragma once

#include <vector>

#include <femx/common/Types.hpp>
#include <femx/eq/TimeResidualEquation.hpp>
#include <femx/eq/TimeStateSolver.hpp>
#include <femx/eq/TimeStateTrajectory.hpp>
#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/inverse/TimeObjectiveFunctional.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/LinearSolver.hpp>

namespace femx
{
namespace inverse
{

/**
 * @brief Reduced objective for time-marching equations using a discrete adjoint.
 *
 * The residual convention is R_k(u_{k+1}, u_k, m) = 0 for
 * k = 0, ..., N - 1. The initial state u_0 is assumed fixed with respect to
 * the reduced parameters.
 */
class TimeAdjointReducedFunctional final : public ReducedFunctional
{
public:
  TimeAdjointReducedFunctional(eq::TimeStateSolver&            state_solver,
                               const eq::TimeResidualEquation& eq,
                               system::LinearSolver&           adjoint_solver,
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
                eq::TimeStateTrajectory& tr);

  void gradAt(const eq::TimeStateTrajectory& tr,
              const Vector<Real>&            prm,
              Vector<Real>&                  out);

  static void checkSize(const Vector<Real>& value,
                        Index               expected,
                        const char*         message);

private:
  eq::TimeStateSolver&            state_solver_;
  const eq::TimeResidualEquation& eq_;
  system::LinearSolver&           adj_solver_;
  const TimeObjectiveFunctional&  obj_;
};

} // namespace inverse
} // namespace femx
