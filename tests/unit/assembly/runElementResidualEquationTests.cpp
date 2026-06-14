#include <iostream>

#include <femx/assembly/DofLayout.hpp>
#include <femx/assembly/ElementKernel.hpp>
#include <femx/assembly/ElementResidualEquation.hpp>
#include <femx/assembly/SparsityPatternBuilder.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Mesh.hpp>
#include <femx/system/native/DenseSystemMatrix.hpp>
#include <femx/system/native/SparseSystemMatrix.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class LinearElementKernel final : public assembly::ElementKernel
{
public:
  void res(Index         ic,
           const Vector& u,
           const Vector& m,
           Vector&       out) const override
  {
    resize(out, u.size());
    const Real scale = stateScale(ic);
    for (Index i = 0; i < u.size(); ++i)
    {
      out[i] = scale * u[i] + 2.0 * m[i];
    }
  }

  void stateJac(Index         ic,
                const Vector& u,
                const Vector& m,
                DenseMatrix&  out) const override
  {
    (void) m;
    out.resize(u.size(), u.size());
    const Real scale = stateScale(ic);
    for (Index i = 0; i < u.size(); ++i)
    {
      out(i, i) = scale;
    }
  }

  void paramJac(Index         ic,
                const Vector& u,
                const Vector& m,
                DenseMatrix&  out) const override
  {
    (void) ic;
    out.resize(u.size(), m.size());
    for (Index i = 0; i < u.size(); ++i)
    {
      out(i, i) = 2.0;
    }
  }

private:
  static Real stateScale(Index ic)
  {
    return 1.0 + static_cast<Real>(ic);
  }

  static void resize(Vector& out, Index size)
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

class ElementResidualEquationTests : public TestBase
{
public:
  TestOutcome assemblesResidualFromElementKernel()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(2, 1);
    LagrangeQuadQ1 elem;
    FESpace        space(&mesh, &elem);
    space.setup();

    LinearElementKernel               kernel;
    assembly::ElementResidualEquation equation{
        assembly::DofLayout(space),
        assembly::DofLayout(space),
        kernel};

    Vector state(space.numDofs());
    Vector params(space.numDofs());
    fillStateAndParams(state, params);

    Vector res;
    equation.res(state, params, res);

    Vector      expected(space.numDofs());
    DenseMatrix expected_state_jac(space.numDofs(), space.numDofs());
    DenseMatrix expected_param_jac(space.numDofs(), space.numDofs());
    assembleExpected(space, state, params, expected, expected_state_jac, expected_param_jac);

    status *= (equation.numStates() == space.numDofs());
    status *= (equation.numParams() == space.numDofs());
    status *= (equation.numRes() == space.numDofs());
    for (Index i = 0; i < space.numDofs(); ++i)
    {
      status *= isEqual(res[i], expected[i]);
    }

    return status.report(__func__);
  }

  TestOutcome assemblesDenseStateJacobian()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(2, 1);
    LagrangeQuadQ1 elem;
    FESpace        space(&mesh, &elem);
    space.setup();

    LinearElementKernel               kernel;
    assembly::ElementResidualEquation equation{
        assembly::DofLayout(space),
        assembly::DofLayout(space),
        kernel};

    Vector state(space.numDofs());
    Vector params(space.numDofs());
    fillStateAndParams(state, params);

    Vector      expected_res(space.numDofs());
    DenseMatrix expected_state_jac(space.numDofs(), space.numDofs());
    DenseMatrix expected_param_jac(space.numDofs(), space.numDofs());
    assembleExpected(space,
                     state,
                     params,
                     expected_res,
                     expected_state_jac,
                     expected_param_jac);

    system::DenseSystemMatrix state_jac;
    equation.assembleStateJac(state, params, state_jac);
    state_jac.finalize();

    for (Index i = 0; i < space.numDofs(); ++i)
    {
      for (Index j = 0; j < space.numDofs(); ++j)
      {
        status *= isEqual(state_jac.matrix()(i, j),
                          expected_state_jac(i, j));
      }
    }

    Vector dir(space.numDofs());
    for (Index i = 0; i < dir.size(); ++i)
    {
      dir[i] = 0.25 * static_cast<Real>(i + 1);
    }

    Vector applied;
    equation.applyStateJac(state, params, dir, applied);
    checkMatVec(status, expected_state_jac, dir, applied);

    equation.applyStateJacT(state, params, dir, applied);
    checkMatTVec(status, expected_state_jac, dir, applied);

    return status.report(__func__);
  }

  TestOutcome assemblesSparseParamJacobian()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(2, 1);
    LagrangeQuadQ1 elem;
    FESpace        space(&mesh, &elem);
    space.setup();

    LinearElementKernel               kernel;
    assembly::ElementResidualEquation equation{
        assembly::DofLayout(space),
        assembly::DofLayout(space),
        kernel};

    Vector state(space.numDofs());
    Vector params(space.numDofs());
    fillStateAndParams(state, params);

    Vector      expected_res(space.numDofs());
    DenseMatrix expected_state_jac(space.numDofs(), space.numDofs());
    DenseMatrix expected_param_jac(space.numDofs(), space.numDofs());
    assembleExpected(space,
                     state,
                     params,
                     expected_res,
                     expected_state_jac,
                     expected_param_jac);

    auto                       pattern = assembly::SparsityPatternBuilder::build(space);
    system::SparseSystemMatrix param_jac(pattern);
    equation.assembleParamJac(state, params, param_jac);
    param_jac.finalize();

    Vector dir(space.numDofs());
    for (Index i = 0; i < dir.size(); ++i)
    {
      dir[i] = -0.5 + 0.1 * static_cast<Real>(i);
    }

    Vector applied;
    param_jac.apply(dir, applied);
    checkMatVec(status, expected_param_jac, dir, applied);

    param_jac.applyT(dir, applied);
    checkMatTVec(status, expected_param_jac, dir, applied);

    return status.report(__func__);
  }

private:
  static void fillStateAndParams(Vector& state, Vector& params)
  {
    for (Index i = 0; i < state.size(); ++i)
    {
      state[i]  = 1.0 + static_cast<Real>(i);
      params[i] = 0.1 * static_cast<Real>(i + 1);
    }
  }

  static void assembleExpected(const FESpace& space,
                               const Vector&  state,
                               const Vector&  params,
                               Vector&        res,
                               DenseMatrix&   state_jac,
                               DenseMatrix&   param_jac)
  {
    res.setZero();
    state_jac.setZero();
    param_jac.setZero();

    for (Index ic = 0; ic < space.numElems(); ++ic)
    {
      const auto dofs  = space.elemDofs(ic);
      const Real scale = 1.0 + static_cast<Real>(ic);
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const Index row      = dofs[i];
        res[row]            += scale * state[row] + 2.0 * params[row];
        state_jac(row, row) += scale;
        param_jac(row, row) += 2.0;
      }
    }
  }

  void checkMatVec(TestStatus&        status,
                   const DenseMatrix& mat,
                   const Vector&      dir,
                   const Vector&      applied)
  {
    for (Index i = 0; i < mat.rows(); ++i)
    {
      Real expected = 0.0;
      for (Index j = 0; j < mat.cols(); ++j)
      {
        expected += mat(i, j) * dir[j];
      }
      status *= isEqual(applied[i], expected);
    }
  }

  void checkMatTVec(TestStatus&        status,
                    const DenseMatrix& mat,
                    const Vector&      dir,
                    const Vector&      applied)
  {
    for (Index j = 0; j < mat.cols(); ++j)
    {
      Real expected = 0.0;
      for (Index i = 0; i < mat.rows(); ++i)
      {
        expected += mat(i, j) * dir[i];
      }
      status *= isEqual(applied[j], expected);
    }
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running elem residual equation tests:\n";

  femx::tests::ElementResidualEquationTests test;

  femx::tests::TestingResults result;
  result += test.assemblesResidualFromElementKernel();
  result += test.assemblesDenseStateJacobian();
  result += test.assemblesSparseParamJacobian();

  return result.summary();
}
