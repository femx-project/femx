#include <array>
#include <cmath>
#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/assembly/Assembler.hpp>
#include <femx/assembly/ElementKernel.hpp>
#include <femx/assembly/FEMResidual.hpp>
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
  status                       *= dims.num_params == 0;
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

} // namespace
} // namespace tests
} // namespace femx

int main(int, char**)
{
  femx::tests::TestingResults results;

  results += femx::tests::assemblerScattersElementVectors();
  results += femx::tests::assemblerScattersElementMatrices();
  results += femx::tests::femResidualAssemblesResidualAndJacobian();

  return results.summary();
}
