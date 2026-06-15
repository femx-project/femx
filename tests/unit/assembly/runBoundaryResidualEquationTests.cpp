#include <iostream>
#include <stdexcept>

#include <femx/assembly/BoundaryDofLayout.hpp>
#include <femx/assembly/BoundaryElementKernel.hpp>
#include <femx/assembly/BoundaryResidualEquation.hpp>
#include <femx/eq/AssembledResidualEquation.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Mesh.hpp>
#include <femx/system/native/DenseSystemMatrix.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

class ZeroAssembledEquation final : public eq::AssembledResidualEquation
{
public:
  explicit ZeroAssembledEquation(Index size)
    : size_(size),
      num_params_(size)
  {
  }

  ZeroAssembledEquation(Index size,
                        Index num_params)
    : size_(size),
      num_params_(num_params)
  {
  }

  Index numStates() const override
  {
    return size_;
  }

  Index numParams() const override
  {
    return num_params_;
  }

  Index numRes() const override
  {
    return size_;
  }

  void res(const Vector& state,
           const Vector& params,
           Vector&       out) const override
  {
    checkSizes(state, params);
    resize(out, size_);
  }

  void assembleStateJac(const Vector&         state,
                        const Vector&         params,
                        system::SystemMatrix& out) const override
  {
    checkSizes(state, params);
    out.resize(size_, size_);
    out.setZero();
  }

  void assembleParamJac(const Vector&         state,
                        const Vector&         params,
                        system::SystemMatrix& out) const override
  {
    checkSizes(state, params);
    out.resize(size_, num_params_);
    out.setZero();
  }

private:
  void checkSizes(const Vector& state, const Vector& params) const
  {
    if (state.size() != size_ || params.size() != num_params_)
    {
      throw std::runtime_error("ZeroAssembledEquation size mismatch");
    }
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

private:
  Index size_{0};
  Index num_params_{0};
};

class LinearBoundaryKernel final : public assembly::BoundaryElementKernel
{
public:
  void res(Index                      ib,
           const Mesh::BoundaryFacet& facet,
           const Vector&              u,
           const Vector&              m,
           Vector&                    out) const override
  {
    (void) facet;
    resize(out, u.size());
    const Real scale = stateScale(ib);
    for (Index i = 0; i < u.size(); ++i)
    {
      out[i] = scale * u[i] + 2.0 * m[i];
    }
  }

  void stateJac(Index                      ib,
                const Mesh::BoundaryFacet& facet,
                const Vector&              u,
                const Vector&              m,
                DenseMatrix&               out) const override
  {
    (void) facet;
    (void) m;
    out.resize(u.size(), u.size());
    const Real scale = stateScale(ib);
    for (Index i = 0; i < u.size(); ++i)
    {
      out(i, i) = scale;
    }
  }

  void paramJac(Index                      ib,
                const Mesh::BoundaryFacet& facet,
                const Vector&              u,
                const Vector&              m,
                DenseMatrix&               out) const override
  {
    (void) ib;
    (void) facet;
    out.resize(u.size(), m.size());
    for (Index i = 0; i < u.size(); ++i)
    {
      out(i, i) = 2.0;
    }
  }

private:
  static Real stateScale(Index ib)
  {
    return 3.0 + static_cast<Real>(ib);
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

class BoundaryResidualEquationTests : public TestBase
{
public:
  TestOutcome addsBoundaryResidualAndJacobians()
  {
    TestStatus status;
    status = true;

    Mesh mesh = makeMeshWithBoundaryFacet();

    LagrangeQuadQ1 elem;
    FESpace        space(&mesh, &elem);
    space.setup();

    ZeroAssembledEquation base(space.numDofs());
    LinearBoundaryKernel  kernel;

    assembly::BoundaryDofLayout        boundary_layout(space, kBoundaryTag);
    assembly::BoundaryResidualEquation equation(
        base, boundary_layout, boundary_layout, boundary_layout, kernel);

    Vector state(space.numDofs());
    Vector params(space.numDofs());
    fillStateAndParams(state, params);

    Vector res;
    equation.res(state, params, res);

    status *= isEqual(res[0], 3.0 * state[0] + 2.0 * params[0]);
    status *= isEqual(res[1], 3.0 * state[1] + 2.0 * params[1]);
    status *= isEqual(res[2], 0.0);
    status *= isEqual(res[3], 0.0);

    system::DenseSystemMatrix state_jac;
    equation.assembleStateJac(state, params, state_jac);
    state_jac.finalize();

    status *= isEqual(state_jac.matrix()(0, 0), 3.0);
    status *= isEqual(state_jac.matrix()(1, 1), 3.0);
    status *= isEqual(state_jac.matrix()(2, 2), 0.0);
    status *= isEqual(state_jac.matrix()(3, 3), 0.0);

    system::DenseSystemMatrix param_jac;
    equation.assembleParamJac(state, params, param_jac);
    param_jac.finalize();

    status *= isEqual(param_jac.matrix()(0, 0), 2.0);
    status *= isEqual(param_jac.matrix()(1, 1), 2.0);
    status *= isEqual(param_jac.matrix()(2, 2), 0.0);
    status *= isEqual(param_jac.matrix()(3, 3), 0.0);

    return status.report(__func__);
  }

  TestOutcome supportsCompactBoundaryParameterLayout()
  {
    TestStatus status;
    status = true;

    Mesh mesh = makeMeshWithTwoBoundaryFacets();

    LagrangeQuadQ1 elem;
    FESpace        space(&mesh, &elem);
    space.setup();

    assembly::BoundaryDofLayout state_layout(space, kBoundaryTag);
    assembly::BoundaryDofLayout param_layout =
        assembly::BoundaryDofLayout::compact(space, kBoundaryTag);

    status *= (state_layout.numDofs() == space.numDofs());
    status *= (param_layout.numDofs() == 3);

    ZeroAssembledEquation              base(space.numDofs(), param_layout.numDofs());
    LinearBoundaryKernel               kernel;
    assembly::BoundaryResidualEquation equation(
        base, state_layout, state_layout, param_layout, kernel);

    Vector state(space.numDofs());
    for (Index i = 0; i < state.size(); ++i)
    {
      state[i] = 1.0 + static_cast<Real>(i);
    }

    Vector params(param_layout.numDofs());
    params[0] = -1.0;
    params[1] = 2.0;
    params[2] = 4.0;

    Vector res;
    equation.res(state, params, res);

    status *= isEqual(res[3], 3.0 * state[3] + 2.0 * params[0]);
    status *= isEqual(res[4],
                      3.0 * state[4] + 2.0 * params[1]
                          + 4.0 * state[4] + 2.0 * params[1]);
    status *= isEqual(res[5], 4.0 * state[5] + 2.0 * params[2]);

    system::DenseSystemMatrix param_jac;
    equation.assembleParamJac(state, params, param_jac);
    param_jac.finalize();

    status *= (param_jac.numCols() == 3);
    status *= isEqual(param_jac.matrix()(3, 0), 2.0);
    status *= isEqual(param_jac.matrix()(4, 1), 4.0);
    status *= isEqual(param_jac.matrix()(5, 2), 2.0);

    return status.report(__func__);
  }

private:
  static Mesh makeMeshWithBoundaryFacet()
  {
    Mesh mesh = Mesh::makeStructuredQuad(1, 1);

    Mesh::BoundaryFacet facet;
    facet.dim           = 1;
    facet.entity_tag    = kBoundaryTag;
    facet.physical_tag  = kBoundaryTag;
    facet.physical_name = "bottom";
    facet.shape         = Cell::Shape::Segment;
    facet.node_ids      = {0, 1};
    mesh.addBoundaryFacet(facet);

    return mesh;
  }

  static Mesh makeMeshWithTwoBoundaryFacets()
  {
    Mesh mesh = Mesh::makeStructuredQuad(2, 1);

    Mesh::BoundaryFacet first;
    first.dim           = 1;
    first.entity_tag    = kBoundaryTag;
    first.physical_tag  = kBoundaryTag;
    first.physical_name = "top";
    first.shape         = Cell::Shape::Segment;
    first.node_ids      = {3, 4};
    mesh.addBoundaryFacet(first);

    Mesh::BoundaryFacet second = first;
    second.node_ids            = {4, 5};
    mesh.addBoundaryFacet(second);

    return mesh;
  }

  static void fillStateAndParams(Vector& state, Vector& params)
  {
    for (Index i = 0; i < state.size(); ++i)
    {
      state[i]  = 1.0 + static_cast<Real>(i);
      params[i] = -0.25 + 0.1 * static_cast<Real>(i);
    }
  }

private:
  static constexpr Index kBoundaryTag = 7;
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running boundary residual equation tests:\n";

  femx::tests::BoundaryResidualEquationTests test;

  femx::tests::TestingResults result;
  result += test.addsBoundaryResidualAndJacobians();
  result += test.supportsCompactBoundaryParameterLayout();

  return result.summary();
}
