#include <iostream>
#include <stdexcept>

#include <femx/problem/LeastSquaresObjective.hpp>
#include <femx/problem/ObservationOperator.hpp>
#include <femx/algebra/Vector.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class LinearObservation final : public problem::ObservationOperator
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
               const Vector<Real>& prm,
               Vector<Real>&       out) const override
  {
    resize(out, numObservations());
    out[0] = 2.0 * state[0] - state[1] + 3.0 * prm[0];
    out[1] = -4.0 * state[0] + 5.0 * state[1] + 7.0 * prm[1];
  }

  void applyStateJac(const Vector<Real>& state,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
    resize(out, numObservations());
    out[0] = 2.0 * dir[0] - dir[1];
    out[1] = -4.0 * dir[0] + 5.0 * dir[1];
  }

  void applyStateJacT(const Vector<Real>& state,
                      const Vector<Real>& prm,
                      const Vector<Real>& dir,
                      Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
    resize(out, numStates());
    out[0] = 2.0 * dir[0] - 4.0 * dir[1];
    out[1] = -dir[0] + 5.0 * dir[1];
  }

  void applyParamJac(const Vector<Real>& state,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
    resize(out, numObservations());
    out[0] = 3.0 * dir[0];
    out[1] = 7.0 * dir[1];
  }

  void applyParamJacT(const Vector<Real>& state,
                      const Vector<Real>& prm,
                      const Vector<Real>& dir,
                      Vector<Real>&       out) const override
  {
    (void) state;
    (void) prm;
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

    LinearObservation obs;

    Vector<Real> data(2);
    data[0] = 0.5;
    data[1] = -1.0;

    const problem::LeastSquaresObjective obj(obs, data, 2.0);

    Vector<Real> state(2);
    state[0] = 0.25;
    state[1] = -0.5;

    Vector<Real> prm(2);
    prm[0] = 1.0;
    prm[1] = -2.0;

    status *= (obj.numStates() == 2);
    status *= (obj.numParams() == 2);
    status *= isEqual(obj.value(state, prm), 284.5);

    Vector<Real> grad;
    obj.stateGrad(state, prm, grad);
    status *= isEqual(grad[0], 146.0);
    status *= isEqual(grad[1], -172.0);

    obj.paramGrad(state, prm, grad);
    status *= isEqual(grad[0], 21.0);
    status *= isEqual(grad[1], -231.0);

    return status.report(__func__);
  }

  TestOutcome constructorRejectsWrongDataSize()
  {
    TestStatus status;
    status = true;

    LinearObservation obs;

    Vector<Real> data(1);
    data[0] = 0.5;

    bool threw = false;
    try
    {
      const problem::LeastSquaresObjective obj(obs, data);
      (void) obj;
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
