#include <iostream>
#include <stdexcept>

#include <femx/inverse/LeastSquaresObjective.hpp>
#include <femx/inverse/ObservationOperator.hpp>
#include <femx/linalg/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class LinearObservation final : public inverse::ObservationOperator
{
public:
  Index numStates() const override
  {
    return 2;
  }

  Index numParams() const override
  {
    return 2;
  }

  Index numObservations() const override
  {
    return 2;
  }

  void observe(const Vector<Real>& state,
               const Vector<Real>& params,
               Vector<Real>&       out) const override
  {
    resize(out, numObservations());
    out[0] = 2.0 * state[0] - state[1] + 3.0 * params[0];
    out[1] = -4.0 * state[0] + 5.0 * state[1] + 7.0 * params[1];
  }

  void applyStateJac(const Vector<Real>& state,
                     const Vector<Real>& params,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numObservations());
    out[0] = 2.0 * dir[0] - dir[1];
    out[1] = -4.0 * dir[0] + 5.0 * dir[1];
  }

  void applyStateJacT(const Vector<Real>& state,
                      const Vector<Real>& params,
                      const Vector<Real>& dir,
                      Vector<Real>&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numStates());
    out[0] = 2.0 * dir[0] - 4.0 * dir[1];
    out[1] = -dir[0] + 5.0 * dir[1];
  }

  void applyParamJac(const Vector<Real>& state,
                     const Vector<Real>& params,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numObservations());
    out[0] = 3.0 * dir[0];
    out[1] = 7.0 * dir[1];
  }

  void applyParamJacT(const Vector<Real>& state,
                      const Vector<Real>& params,
                      const Vector<Real>& dir,
                      Vector<Real>&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numParams());
    out[0] = 3.0 * dir[0];
    out[1] = 7.0 * dir[1];
  }

private:
  static void resize(Vector<Real>& out, Index size)
  {
    if (out.size() != size)
    {
      out.resize(size);
    }
    else
    {
      out.setZero();
    }
  }
};

class LeastSquaresObjectiveTests : public TestBase
{
public:
  TestOutcome valueAndGradientsUseObservationTranspose()
  {
    TestStatus status;
    status = true;

    LinearObservation observation;

    Vector<Real> data(2);
    data[0] = 0.5;
    data[1] = -1.0;

    const inverse::LeastSquaresObjective objective(observation, data, 2.0);

    Vector<Real> state(2);
    state[0] = 0.25;
    state[1] = -0.5;

    Vector<Real> params(2);
    params[0] = 1.0;
    params[1] = -2.0;

    status *= (objective.numStates() == 2);
    status *= (objective.numParams() == 2);
    status *= isEqual(objective.value(state, params), 284.5);

    Vector<Real> grad;
    objective.stateGrad(state, params, grad);
    status *= isEqual(grad[0], 146.0);
    status *= isEqual(grad[1], -172.0);

    objective.paramGrad(state, params, grad);
    status *= isEqual(grad[0], 21.0);
    status *= isEqual(grad[1], -231.0);

    return status.report(__func__);
  }

  TestOutcome constructorRejectsWrongDataSize()
  {
    TestStatus status;
    status = true;

    LinearObservation observation;

    Vector<Real> data(1);
    data[0] = 0.5;

    bool threw = false;
    try
    {
      const inverse::LeastSquaresObjective objective(observation, data);
      (void) objective;
    }
    catch (const std::runtime_error&)
    {
      threw = true;
    }

    status *= threw;
    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running least-squares objective tests:\n";

  femx::tests::LeastSquaresObjectiveTests test;

  femx::tests::TestingResults result;
  result += test.valueAndGradientsUseObservationTranspose();
  result += test.constructorRejectsWrongDataSize();

  return result.summary();
}
