#include <array>
#include <cmath>
#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/assembly/Assembler.hpp>
#include <femx/assembly/DirichletControlResidual.hpp>
#include <femx/assembly/ElementKernel.hpp>
#include <femx/assembly/FEMResidual.hpp>
#include <femx/assembly/TimeDirichletControlResidual.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/native/DenseAssemblyMatrix.hpp>
#include <femx/state/Linearization.hpp>

namespace femx
{
using namespace fem;

namespace tests
{
namespace
{

class AffineElementKernel final : public assembly::ElementKernel
{
public:
  void res(Index,
           const Vector<Real>& u,
           const Vector<Real>&,
           Vector<Real>& out) const override
  {
    out.resize(u.size());
    for (Index i = 0; i < u.size(); ++i)
    {
      out[i] = u[i] + static_cast<Real>(i + 1);
    }
  }

  void stateJac(Index,
                const Vector<Real>& u,
                const Vector<Real>&,
                DenseMatrix& out) const override
  {
    out.resize(u.size(), u.size());
    out.setZero();
    for (Index i = 0; i < u.size(); ++i)
    {
      out(i, i) = 2.0;
    }
  }

  void paramJac(Index,
                const Vector<Real>& u,
                const Vector<Real>&,
                DenseMatrix& out) const override
  {
    out.resize(u.size(), 0);
  }
};

class IdentityResidual final : public state::Residual
{
public:
  state::Dimensions dims() const override
  {
    return {3, 0, 3};
  }

  void res(const Vector<Real>& state,
           const Vector<Real>&,
           Vector<Real>& out) const override
  {
    out = state;
  }

  void linearize(const Vector<Real>&,
                 const Vector<Real>&,
                 state::Linearization& out) const override
  {
    auto* matrices = dynamic_cast<state::MatrixLinearization*>(&out);
    if (matrices == nullptr)
    {
      throw std::runtime_error("IdentityResidual requires matrices");
    }

    auto& state_jac = matrices->stateMat();
    state_jac.resize(3, 3);
    state_jac.setZero();
    for (Index i = 0; i < 3; ++i)
    {
      state_jac.set(i, i, 1.0);
    }
    state_jac.finalize();

    auto& param_jac = matrices->paramMat();
    param_jac.resize(3, 0);
    param_jac.setZero();
    param_jac.finalize();
  }
};

class IdentityTimeResidual final : public state::TimeResidual
{
public:
  state::TimeDims dims() const override
  {
    return {2, 3, 0, 3, 1};
  }

  void res(const state::TimeContext& ctx,
           Vector<Real>&             out) const override
  {
    out = *ctx.nxt;
  }

  void applyJac(const state::TimeContext&,
                state::VariableBlock wrt,
                const Vector<Real>&  dir,
                Vector<Real>&        out) const override
  {
    resizeOrZero(out, 3);
    if (wrt.isNextState())
    {
      out = dir;
    }
  }

  void applyJacT(const state::TimeContext&,
                 state::VariableBlock wrt,
                 const Vector<Real>&  adj,
                 Vector<Real>&        out) const override
  {
    if (wrt.isParam())
    {
      out.resize(0);
      return;
    }
    out = adj;
  }

  bool assembleJac(const state::TimeContext&,
                   state::VariableBlock   wrt,
                   linalg::MatrixBuilder& out) const override
  {
    const Index cols = wrt.isParam() ? 0 : 3;
    out.resize(3, cols);
    out.setZero();
    if (wrt.isNextState())
    {
      for (Index i = 0; i < 3; ++i)
      {
        out.set(i, i, 1.0);
      }
    }
    return true;
  }
};

bool near(Real a, Real b)
{
  return std::abs(a - b) <= 1.0e-12;
}

template <std::size_t N>
bool valuesNear(const Vector<Real>&        actual,
                const std::array<Real, N>& expected)
{
  if (actual.size() != static_cast<Index>(N))
  {
    return false;
  }
  for (std::size_t i = 0; i < N; ++i)
  {
    if (!near(actual[static_cast<Index>(i)], expected[i]))
    {
      return false;
    }
  }
  return true;
}

DenseMatrix constantLocalMatrix(Index size, Real value)
{
  DenseMatrix local(size, size);
  for (Index i = 0; i < size; ++i)
  {
    for (Index j = 0; j < size; ++j)
    {
      local(i, j) = value;
    }
  }
  return local;
}

FESpace makeSpace(Mesh& mesh, LagrangeQuadQ1& element)
{
  FESpace space(&mesh, &element);
  space.setup();
  return space;
}

TestOutcome assemblerScattersElementVectors()
{
  TestStatus status(__func__);

  Mesh           mesh = Mesh::makeStructuredQuad(2, 1);
  LagrangeQuadQ1 element;
  FESpace        space = makeSpace(mesh, element);

  assembly::Assembler assembler(space);
  Vector<Real>        out;
  assembler.initVec(out);

  assembler.addVec(0, Vector<Real>{1.0, 2.0, 3.0, 4.0}, out);
  assembler.addVec(1, Vector<Real>{10.0, 20.0, 30.0, 40.0}, out);

  status *= valuesNear(out,
                       std::array<Real, 6>{{1.0, 12.0, 20.0, 4.0, 43.0, 30.0}});

  bool threw = false;
  try
  {
    assembler.addVec(0, Vector<Real>{1.0, 2.0}, out);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

TestOutcome assemblerScattersElementMatrices()
{
  TestStatus status(__func__);

  Mesh           mesh = Mesh::makeStructuredQuad(2, 1);
  LagrangeQuadQ1 element;
  FESpace        space = makeSpace(mesh, element);

  assembly::Assembler         assembler(space);
  linalg::DenseAssemblyMatrix matrix;
  assembler.initMat(matrix);

  assembler.addMat(0, constantLocalMatrix(4, 1.0), matrix);
  assembler.addMat(1, constantLocalMatrix(4, 2.0), matrix);

  status *= matrix.numRows() == 6;
  status *= matrix.numCols() == 6;
  status *= near(matrix.mat()(0, 0), 1.0);
  status *= near(matrix.mat()(0, 3), 1.0);
  status *= near(matrix.mat()(1, 4), 3.0);
  status *= near(matrix.mat()(2, 5), 2.0);
  status *= near(matrix.mat()(0, 2), 0.0);

  bool threw = false;
  try
  {
    assembler.addMat(0, constantLocalMatrix(3, 1.0), matrix);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

TestOutcome femResidualAssemblesResidualAndJacobian()
{
  TestStatus status(__func__);

  Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
  LagrangeQuadQ1 element;
  FESpace        space = makeSpace(mesh, element);

  const AffineElementKernel kernel;
  assembly::FEMResidual     residual(DofLayout(space), kernel);

  const state::Dimensions dims  = residual.dims();
  status                       *= dims.num_states == 4;
  status                       *= dims.num_param == 0;
  status                       *= dims.num_residuals == 4;

  Vector<Real> out;
  residual.res(Vector<Real>{10.0, 20.0, 30.0, 40.0}, Vector<Real>{}, out);
  status *= valuesNear(out,
                       std::array<Real, 4>{{11.0, 22.0, 34.0, 43.0}});

  linalg::DenseAssemblyMatrix state_jac;
  linalg::DenseAssemblyMatrix param_jac;
  state::MatrixLinearization  lin(state_jac, param_jac);
  residual.linearize(Vector<Real>{10.0, 20.0, 30.0, 40.0},
                     Vector<Real>{},
                     lin);

  status *= state_jac.numRows() == 4;
  status *= state_jac.numCols() == 4;
  status *= param_jac.numRows() == 4;
  status *= param_jac.numCols() == 0;
  status *= near(state_jac.mat()(0, 0), 2.0);
  status *= near(state_jac.mat()(1, 1), 2.0);
  status *= near(state_jac.mat()(0, 1), 0.0);

  bool threw = false;
  try
  {
    residual.res(Vector<Real>{1.0}, Vector<Real>{}, out);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  status *= threw;

  return status.report();
}

fem::DirichletControl mappedControl()
{
  return fem::DirichletControl(
      Vector<Index>{0, 2},
      1,
      Vector<fem::DirichletControlMapEntry>{{0, 0, 2.0},
                                            {1, 0, -1.0}});
}

TestOutcome mappedStationaryDirichletResidual()
{
  TestStatus status(__func__);

  const IdentityResidual                   base;
  const assembly::DirichletControlResidual residual(base, mappedControl());

  status *= residual.dims().num_param == 1;

  Vector<Real> out;
  residual.res(Vector<Real>{10.0, 20.0, 30.0},
               Vector<Real>{3.0},
               out);
  status *= valuesNear(out, std::array<Real, 3>{{4.0, 20.0, 33.0}});

  linalg::DenseAssemblyMatrix state_jac;
  linalg::DenseAssemblyMatrix param_jac;
  state::MatrixLinearization  linearization(state_jac, param_jac);
  residual.linearize(Vector<Real>{10.0, 20.0, 30.0},
                     Vector<Real>{3.0},
                     linearization);

  status *= param_jac.numRows() == 3;
  status *= param_jac.numCols() == 1;
  status *= near(param_jac.mat()(0, 0), -2.0);
  status *= near(param_jac.mat()(1, 0), 0.0);
  status *= near(param_jac.mat()(2, 0), 1.0);

  return status.report();
}

TestOutcome mappedTimeDirichletResidual()
{
  TestStatus status(__func__);

  const IdentityTimeResidual                   base;
  const assembly::TimeDirichletControlResidual residual(
      base,
      mappedControl(),
      {},
      0,
      -1,
      {},
      Vector<LinearInterpolation>{{0, 1, 0.25}, {1, 1, 0.0}});

  status *= residual.numParams() == 2;

  const Vector<Real>       history{1.0, 2.0, 3.0};
  const Vector<Real>       next{10.0, 20.0, 30.0};
  const Vector<Real>       parameters{4.0, 8.0};
  const state::TimeContext ctx{
      0,
      &next,
      &parameters,
      state::TimeHistoryView(history.data(), 1, 3)};

  Vector<Real> out;
  residual.res(ctx, out);
  status *= valuesNear(out, std::array<Real, 3>{{0.0, 20.0, 35.0}});

  residual.applyJac(
      ctx, state::VariableBlock::Param, Vector<Real>{2.0, 6.0}, out);
  status *= valuesNear(out, std::array<Real, 3>{{-6.0, 0.0, 3.0}});

  residual.applyJacT(
      ctx, state::VariableBlock::Param, Vector<Real>{1.0, 5.0, 3.0}, out);
  status *= valuesNear(out, std::array<Real, 2>{{0.75, 0.25}});

  linalg::DenseAssemblyMatrix param_jac;
  status *= residual.assembleJac(
      ctx, state::VariableBlock::Param, param_jac);
  status *= param_jac.numRows() == 3;
  status *= param_jac.numCols() == 2;
  status *= near(param_jac.mat()(0, 0), -1.5);
  status *= near(param_jac.mat()(0, 1), -0.5);
  status *= near(param_jac.mat()(1, 0), 0.0);
  status *= near(param_jac.mat()(1, 1), 0.0);
  status *= near(param_jac.mat()(2, 0), 0.75);
  status *= near(param_jac.mat()(2, 1), 0.25);

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main(int, char**)
{
  femx::tests::TestingResults results;

  results += femx::tests::assemblerScattersElementVectors();
  results += femx::tests::assemblerScattersElementMatrices();
  results += femx::tests::femResidualAssemblesResidualAndJacobian();
  results += femx::tests::mappedStationaryDirichletResidual();
  results += femx::tests::mappedTimeDirichletResidual();

  return results.summary();
}
