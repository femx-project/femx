#include <stdexcept>

#include "../state/StationaryFixtures.hpp"
#include "TestHelper.hpp"
#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/inverse/TimeBlockRegularization.hpp>
#include <femx/inverse/TimeLeastSquaresObjective.hpp>
#include <femx/inverse/TimeObservationData.hpp>
#include <femx/inverse/TimeObservationOperator.hpp>
#include <femx/linalg/Dense.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/StateSolver.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace tests
{
namespace
{

class LinearTimeObservation final : public inverse::TimeObservationOperator
{
public:
  Index numSteps() const override
  {
    return 1;
  }

  Index numStates() const override
  {
    return 2;
  }

  Index numParams() const override
  {
    return 1;
  }

  Index numObservations() const override
  {
    return 2;
  }

  void observe(Index,
               const HostVector& state,
               const HostVector& prm,
               HostVector&       out) const override
  {
    out = {state[0] + prm[0], 2.0 * state[1]};
  }

  void applyStateJac(Index,
                     const HostVector&,
                     const HostVector&,
                     const HostVector& dir,
                     HostVector&       out) const override
  {
    out = {dir[0], 2.0 * dir[1]};
  }

  void applyStateJacT(Index,
                      const HostVector&,
                      const HostVector&,
                      const HostVector& dir,
                      HostVector&       out) const override
  {
    out = {dir[0], 2.0 * dir[1]};
  }

  void applyParamJac(Index,
                     const HostVector&,
                     const HostVector&,
                     const HostVector& dir,
                     HostVector&       out) const override
  {
    out = {dir[0], 0.0};
  }

  void applyParamJacT(Index,
                      const HostVector&,
                      const HostVector&,
                      const HostVector& dir,
                      HostVector&       out) const override
  {
    out = {dir[0]};
  }
};

TestOutcome timeLeastSquaresUsesObservationWeights()
{
  TestStatus status(__func__);

  LinearTimeObservation        observation;
  inverse::TimeObservationData data(1, 2);
  data[0][0] = 1.5;
  data[0][1] = 5.0;
  data.setTimeValues({0.25});

  inverse::TimeLeastSquaresObjective objective(
      observation,
      data,
      {1.0, 1.0},
      {4.0, 9.0},
      0.5);

  state::TimeTrajectory trajectory(1, 2);
  trajectory[0][0] = 1.0;
  trajectory[0][1] = 2.0;
  trajectory[1][0] = 3.0;
  trajectory[1][1] = 4.0;
  const HostVector parameters{0.5};

  status *= std::abs(objective.value(trajectory, parameters) - 6.5)
            < 1.0e-14;

  HostVector state_gradient;
  objective.stateGrad(0, trajectory, parameters, state_gradient);
  status *= state_gradient.size() == 2;
  status *= std::abs(state_gradient[0] - 2.0) < 1.0e-14;
  status *= std::abs(state_gradient[1] - 9.0) < 1.0e-14;
  objective.stateGrad(1, trajectory, parameters, state_gradient);
  status *= std::abs(state_gradient[0] - 2.0) < 1.0e-14;
  status *= std::abs(state_gradient[1] - 9.0) < 1.0e-14;

  HostVector param_gradient;
  objective.paramGrad(trajectory, parameters, param_gradient);
  status *= param_gradient.size() == 1;
  status *= std::abs(param_gradient[0] - 4.0) < 1.0e-14;

  return status.report();
}

TestOutcome timeLeastSquaresRejectsInvalidObservationWeights()
{
  TestStatus status(__func__);

  LinearTimeObservation        observation;
  inverse::TimeObservationData data(1, 2);
  data.setTimeValues({0.25});

  bool threw = false;
  try
  {
    inverse::TimeLeastSquaresObjective objective(
        observation,
        data,
        {1.0, 1.0},
        {1.0, -1.0},
        0.5);
    (void) objective;
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

TestOutcome timeBlockRegularizationUsesSparseQuadraticForm()
{
  TestStatus status(__func__);

  inverse::TimeBlockRegularization objective(
      1,
      2,
      2,
      2,
      {0, 0, 1, 1},
      {0, 1, 0, 1},
      {2.0, -1.0, -1.0, 3.0},
      2.0,
      {1.0, 1.0, 0.0, 0.0});
  state::TimeTrajectory trajectory(1, 2);
  const HostVector      param{2.0, 0.0, 1.0, 2.0};

  status *= std::abs(objective.value(trajectory, param) - 17.0)
            < 1.0e-14;
  HostVector grad;
  objective.paramGrad(trajectory, param, grad);
  status *= grad.size() == 4;
  status *= std::abs(grad[0] - 6.0) < 1.0e-14;
  status *= std::abs(grad[1] + 8.0) < 1.0e-14;
  status *= std::abs(grad[2]) < 1.0e-14;
  status *= std::abs(grad[3] - 10.0) < 1.0e-14;

  return status.report();
}

TestOutcome hostBackendRunsStationarySolversAndAdjoint()
{
  TestStatus status(__func__);

  stationary::AffineResidual<linalg::HostCsrBackend> res(2.0);
  stationary::QuadraticObjective                     obj(1.0, 0.25);
  HostCsrMatrix                                      fwd_jac(res.graph());
  HostCsrMatrix                                      adj_jac(res.graph());
  linalg::DenseLinearSolver                          fwd_solver;
  linalg::DenseLinearSolver                          adj_solver;
  CpuContext                                         ctx;
  state::LinearStateSolver<linalg::HostCsrBackend>   state_solver(
      res, fwd_jac, fwd_solver, ctx);
  inverse::ReducedFunctional<linalg::HostCsrBackend> reduced(
      state_solver, adj_jac, adj_solver, obj);

  const HostVector prm{0.6};
  HostVector       grad;
  const Real       val  = reduced.valueGrad(prm, grad);
  status               *= std::abs(val - 0.29) < 1.0e-13;
  status               *= grad.size() == 1;
  status               *= std::abs(grad[0] + 0.2) < 1.0e-13;

  constexpr Real   eps = 1.0e-6;
  const HostVector plus{prm[0] + eps};
  const HostVector minus{prm[0] - eps};
  const Real       fd =
      (reduced.value(plus) - reduced.value(minus)) / (2.0 * eps);
  status *= std::abs(grad[0] - fd) < 1.0e-9;

  stationary::QuadraticResidual                    nonlinear_res;
  HostCsrMatrix                                    nonlinear_jac(nonlinear_res.graph());
  linalg::DenseLinearSolver                        nonlinear_solver;
  state::NewtonStateSolver<linalg::HostCsrBackend> newton(
      nonlinear_res, nonlinear_jac, nonlinear_solver, ctx);
  newton.setInitialState(HostVector{1.0});
  HostVector state;
  newton.solve(HostVector{4.0}, state);
  status *= state.size() == 1;
  status *= std::abs(state[0] - 2.0) < 1.0e-10;

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main()
{
  femx::tests::TestingResults results;
  results += femx::tests::timeLeastSquaresUsesObservationWeights();
  results += femx::tests::timeLeastSquaresRejectsInvalidObservationWeights();
  results += femx::tests::timeBlockRegularizationUsesSparseQuadraticForm();
  results += femx::tests::hostBackendRunsStationarySolversAndAdjoint();
  return results.summary();
}
