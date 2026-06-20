#include <iostream>

#include <femx/problem/TimeProblemAdapter.hpp>
#include <femx/algebra/backends/native/DenseSystemMatrix.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class LinearTimeMatrixResidualEquation final
    : public problem::TimeMatrixResidualEquation
{
public:
  Index numSteps() const override
  {
    return 2;
  }

  Index numStates() const override
  {
    return 2;
  }

  Index numParams() const override
  {
    return 1;
  }

  Index numRes() const override
  {
    return 2;
  }

  void res(Index               step,
           const Vector<Real>& x_next,
           const Vector<Real>& x,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override
  {
    resize(out, numRes());
    out[0] = next(0, 0) * x_next[0] + next(0, 1) * x_next[1]
             + prev(0, 0) * x[0] + prev(0, 1) * x[1]
             + param(0, 0) * prm[0] + static_cast<Real>(step);
    out[1] = next(1, 0) * x_next[0] + next(1, 1) * x_next[1]
             + prev(1, 0) * x[0] + prev(1, 1) * x[1]
             + param(1, 0) * prm[0] - static_cast<Real>(step);
  }

  void assembleNextStateJac(Index                 step,
                            const Vector<Real>&   x_next,
                            const Vector<Real>&   x,
                            const Vector<Real>&   prm,
                            algebra::SystemMatrix& out) const override
  {
    (void) step;
    (void) x_next;
    (void) x;
    (void) prm;
    assemble2x2(next(0, 0), next(0, 1), next(1, 0), next(1, 1), out);
  }

  void assemblePrevStateJac(Index                 step,
                            const Vector<Real>&   x_next,
                            const Vector<Real>&   x,
                            const Vector<Real>&   prm,
                            algebra::SystemMatrix& out) const override
  {
    (void) step;
    (void) x_next;
    (void) x;
    (void) prm;
    assemble2x2(prev(0, 0), prev(0, 1), prev(1, 0), prev(1, 1), out);
  }

  void assembleParamJac(Index                 step,
                        const Vector<Real>&   x_next,
                        const Vector<Real>&   x,
                        const Vector<Real>&   prm,
                        algebra::SystemMatrix& out) const override
  {
    (void) step;
    (void) x_next;
    (void) x;
    (void) prm;
    out.resize(numRes(), numParams());
    out.setZero();
    out.set(0, 0, param(0, 0));
    out.set(1, 0, param(1, 0));
  }

private:
  static Real next(Index row, Index col)
  {
    constexpr Real values[4] = {2.0, 3.0, 5.0, 7.0};
    return values[row * 2 + col];
  }

  static Real prev(Index row, Index col)
  {
    constexpr Real values[4] = {11.0, 13.0, 17.0, 19.0};
    return values[row * 2 + col];
  }

  static Real param(Index row, Index)
  {
    constexpr Real values[2] = {23.0, 29.0};
    return values[row];
  }

  static void assemble2x2(Real                  a00,
                          Real                  a01,
                          Real                  a10,
                          Real                  a11,
                          algebra::SystemMatrix& out)
  {
    out.resize(2, 2);
    out.setZero();
    out.set(0, 0, a00);
    out.set(0, 1, a01);
    out.set(1, 0, a10);
    out.set(1, 1, a11);
  }

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

class TimeResidualAdapterTests : public TestBase
{
public:
  TestOutcome operatorAdapterExposesAllVariableBlocks()
  {
    TestStatus status;
    status = true;

    LinearTimeMatrixResidualEquation     eq;
    problem::TimeResidualEquationProblemAdapter adapter(eq);

    const problem::TimeDimensions dims = adapter.dimensions();
    status *= (dims.num_steps == 2);
    status *= (dims.num_states == 2);
    status *= (dims.num_params == 1);
    status *= (dims.num_residuals == 2);

    Vector<Real> previous(2);
    previous[0] = 0.25;
    previous[1] = -0.5;

    Vector<Real> next(2);
    next[0] = 1.5;
    next[1] = -2.0;

    Vector<Real> prm(1);
    prm[0] = 0.75;

    problem::TimeContext ctx;
    ctx.step           = 1;
    ctx.previous_state = &previous;
    ctx.next_state     = &next;
    ctx.prm            = &prm;

    Vector<Real> out;
    adapter.residual(ctx, out);
    status *= isEqual(out[0],
                      2.0 * next[0] + 3.0 * next[1]
                          + 11.0 * previous[0] + 13.0 * previous[1]
                          + 23.0 * prm[0] + 1.0);
    status *= isEqual(out[1],
                      5.0 * next[0] + 7.0 * next[1]
                          + 17.0 * previous[0] + 19.0 * previous[1]
                          + 29.0 * prm[0] - 1.0);

    Vector<Real> dir_state(2);
    dir_state[0] = -3.0;
    dir_state[1] = 4.0;

    adapter.applyJacobian(
        ctx, problem::VariableBlock::NextState, dir_state, out);
    status *= isEqual(out[0], 2.0 * dir_state[0] + 3.0 * dir_state[1]);
    status *= isEqual(out[1], 5.0 * dir_state[0] + 7.0 * dir_state[1]);

    adapter.applyJacobian(
        ctx, problem::VariableBlock::PreviousState, dir_state, out);
    status *= isEqual(out[0], 11.0 * dir_state[0] + 13.0 * dir_state[1]);
    status *= isEqual(out[1], 17.0 * dir_state[0] + 19.0 * dir_state[1]);

    Vector<Real> dir_param(1);
    dir_param[0] = 2.5;

    adapter.applyJacobian(
        ctx, problem::VariableBlock::Parameter, dir_param, out);
    status *= isEqual(out[0], 23.0 * dir_param[0]);
    status *= isEqual(out[1], 29.0 * dir_param[0]);

    Vector<Real> lambda(2);
    lambda[0] = 1.25;
    lambda[1] = -0.75;

    adapter.applyJacobianT(
        ctx, problem::VariableBlock::NextState, lambda, out);
    status *= isEqual(out[0], 2.0 * lambda[0] + 5.0 * lambda[1]);
    status *= isEqual(out[1], 3.0 * lambda[0] + 7.0 * lambda[1]);

    adapter.applyJacobianT(
        ctx, problem::VariableBlock::PreviousState, lambda, out);
    status *= isEqual(out[0], 11.0 * lambda[0] + 17.0 * lambda[1]);
    status *= isEqual(out[1], 13.0 * lambda[0] + 19.0 * lambda[1]);

    adapter.applyJacobianT(
        ctx, problem::VariableBlock::Parameter, lambda, out);
    status *= isEqual(out[0], 23.0 * lambda[0] + 29.0 * lambda[1]);

    return status.report(__func__);
  }

  TestOutcome matrixAdapterAssemblesJacobianBlocks()
  {
    TestStatus status;
    status = true;

    LinearTimeMatrixResidualEquation eq;
    problem::TimeMatrixResidualEquationProblemAdapter adapter(eq);

    Vector<Real> previous(2);
    previous[0] = 0.0;
    previous[1] = 0.0;

    Vector<Real> next(2);
    next[0] = 0.0;
    next[1] = 0.0;

    Vector<Real> prm(1);
    prm[0] = 0.0;

    problem::TimeContext ctx;
    ctx.step           = 0;
    ctx.previous_state = &previous;
    ctx.next_state     = &next;
    ctx.prm            = &prm;

    algebra::DenseSystemMatrix matrix;
    status *= adapter.assembleJacobian(
        ctx, problem::VariableBlock::NextState, matrix);

    Vector<Real> dir(2);
    dir[0] = 6.0;
    dir[1] = -2.0;

    Vector<Real> out;
    matrix.apply(dir, out);
    status *= isEqual(out[0], 2.0 * dir[0] + 3.0 * dir[1]);
    status *= isEqual(out[1], 5.0 * dir[0] + 7.0 * dir[1]);

    status *= adapter.assembleJacobian(
        ctx, problem::VariableBlock::PreviousState, matrix);
    matrix.apply(dir, out);
    status *= isEqual(out[0], 11.0 * dir[0] + 13.0 * dir[1]);
    status *= isEqual(out[1], 17.0 * dir[0] + 19.0 * dir[1]);

    status *= adapter.assembleJacobian(
        ctx, problem::VariableBlock::Parameter, matrix);

    Vector<Real> param_dir(1);
    param_dir[0] = -3.0;
    matrix.apply(param_dir, out);
    status *= isEqual(out[0], 23.0 * param_dir[0]);
    status *= isEqual(out[1], 29.0 * param_dir[0]);

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running time residual adapter tests:\n";

  femx::tests::TimeResidualAdapterTests test;

  femx::tests::TestingResults result;
  result += test.operatorAdapterExposesAllVariableBlocks();
  result += test.matrixAdapterAssemblesJacobianBlocks();

  return result.summary();
}
