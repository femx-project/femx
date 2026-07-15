#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/inverse/TimeBlockRegularization.hpp>
#include <femx/inverse/TimeLeastSquaresObjective.hpp>
#include <femx/inverse/TimeObservationData.hpp>
#include <femx/inverse/TimeObservationOperator.hpp>
#include <femx/linalg/Vector.hpp>
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
               const Vector<Real>& state,
               const Vector<Real>& prm,
               Vector<Real>&       out) const override
  {
    out = {state[0] + prm[0], 2.0 * state[1]};
  }

  void applyStateJac(Index,
                     const Vector<Real>&,
                     const Vector<Real>&,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    out = {dir[0], 2.0 * dir[1]};
  }

  void applyStateJacT(Index,
                      const Vector<Real>&,
                      const Vector<Real>&,
                      const Vector<Real>& dir,
                      Vector<Real>&       out) const override
  {
    out = {dir[0], 2.0 * dir[1]};
  }

  void applyParamJac(Index,
                     const Vector<Real>&,
                     const Vector<Real>&,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    out = {dir[0], 0.0};
  }

  void applyParamJacT(Index,
                      const Vector<Real>&,
                      const Vector<Real>&,
                      const Vector<Real>& dir,
                      Vector<Real>&       out) const override
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
  const Vector<Real> parameters{0.5};

  status *= std::abs(objective.value(trajectory, parameters) - 6.5)
            < 1.0e-14;

  Vector<Real> state_gradient;
  objective.stateGrad(0, trajectory, parameters, state_gradient);
  status *= state_gradient.size() == 2;
  status *= std::abs(state_gradient[0] - 2.0) < 1.0e-14;
  status *= std::abs(state_gradient[1] - 9.0) < 1.0e-14;
  objective.stateGrad(1, trajectory, parameters, state_gradient);
  status *= std::abs(state_gradient[0] - 2.0) < 1.0e-14;
  status *= std::abs(state_gradient[1] - 9.0) < 1.0e-14;

  Vector<Real> param_gradient;
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
  const Vector<Real>    param{2.0, 0.0, 1.0, 2.0};

  status *= std::abs(objective.value(trajectory, param) - 17.0)
            < 1.0e-14;
  Vector<Real> grad;
  objective.paramGrad(trajectory, param, grad);
  status *= grad.size() == 4;
  status *= std::abs(grad[0] - 6.0) < 1.0e-14;
  status *= std::abs(grad[1] + 8.0) < 1.0e-14;
  status *= std::abs(grad[2]) < 1.0e-14;
  status *= std::abs(grad[3] - 10.0) < 1.0e-14;

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
  return results.summary();
}
