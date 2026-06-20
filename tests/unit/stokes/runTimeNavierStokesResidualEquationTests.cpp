#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "NavierStokesEquation.hpp"
#include <femx/assembly/SparsityPatternBuilder.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/problem/TimeDirichletControlEquation.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/algebra/backends/native/DenseSystemMatrix.hpp>
#include <femx/algebra/backends/native/SparseSystemMatrix.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

namespace
{

MixedFESpace makeSpace(Mesh&           mesh,
                       LagrangeQuadQ1& element)
{
  FESpace velocity_space(&mesh, &element, mesh.dim());
  FESpace pressure_space(&mesh, &element);

  MixedFESpace space;
  space.addField(velocity_space);
  space.addField(pressure_space);
  space.setup();
  return space;
}

void addBoundaryFacet(Mesh&                     mesh,
                      Index                     tag,
                      const std::string&        name,
                      const std::vector<Index>& node_ids)
{
  Mesh::BoundaryFacet facet;
  facet.dim           = 1;
  facet.entity_tag    = tag;
  facet.physical_tag  = tag;
  facet.physical_name = name;
  facet.shape         = Cell::Shape::Segment;
  facet.node_ids      = node_ids;
  mesh.addBoundaryFacet(std::move(facet));
}

void fillState(Vector<Real>& state,
               Real          offset,
               Real          slope)
{
  for (Index i = 0; i < state.size(); ++i)
  {
    state[i] = offset + slope * static_cast<Real>(i);
  }
}

bool isClose(Real a,
             Real b,
             Real tolerance)
{
  return std::abs(a - b) <= tolerance;
}

} // namespace

class NavierStokesEquationTests : public TestBase
{
public:
  TestOutcome residualAndJacobiansHaveConsistentSizes()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
    LagrangeQuadQ1 element;
    MixedFESpace   space = makeSpace(mesh, element);

    TimeNavierStokesParameters ns_prm;
    ns_prm.steps     = 2;
    ns_prm.dt        = 0.25;
    ns_prm.fluid.rho = 1.4;
    ns_prm.fluid.mu  = 0.03;

    NavierStokesEquation eq(space, ns_prm);

    Vector<Real> x_next(eq.numStates());
    Vector<Real> x(eq.numStates());
    fillState(x_next, 0.11, 0.013);
    fillState(x, -0.07, 0.017);

    Vector<Real> prm;
    Vector<Real> res;
    eq.res(0, x_next, x, prm, res);

#if defined(FEMX_HAS_ENZYME)
    algebra::DenseSystemMatrix next_jac;
    eq.assembleNextStateJac(
        0, x_next, x, prm, next_jac);

    algebra::DenseSystemMatrix prev_jac;
    eq.assemblePrevStateJac(
        0, x_next, x, prm, prev_jac);
#endif

    status *= (res.size() == eq.numRes());
#if defined(FEMX_HAS_ENZYME)
    status *= (next_jac.numRows() == eq.numRes());
    status *= (next_jac.numCols() == eq.numStates());
    status *= (prev_jac.numRows() == eq.numRes());
    status *= (prev_jac.numCols() == eq.numStates());
#endif

    for (Index i = 0; i < res.size(); ++i)
    {
      status *= std::isfinite(res[i]);
    }

    return status.report(__func__);
  }

  TestOutcome jacobiansMatchFiniteDifferences()
  {
    TestStatus status;
    status = true;

#if !defined(FEMX_HAS_ENZYME)
    status.skipTest();
    return status.report(__func__);
#endif

    Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
    LagrangeQuadQ1 element;
    MixedFESpace   space = makeSpace(mesh, element);

    TimeNavierStokesParameters ns_prm;
    ns_prm.steps     = 2;
    ns_prm.dt        = 0.25;
    ns_prm.fluid.rho = 1.4;
    ns_prm.fluid.mu  = 0.03;

    NavierStokesEquation eq(space, ns_prm);

    Vector<Real> x_next(eq.numStates());
    Vector<Real> x(eq.numStates());
    Vector<Real> next_dir(eq.numStates());
#if defined(FEMX_HAS_ENZYME)
    Vector<Real> prev_dir(eq.numStates());
#endif
    fillState(x_next, 0.11, 0.013);
    fillState(x, -0.07, 0.017);
    fillState(next_dir, 0.03, -0.005);
#if defined(FEMX_HAS_ENZYME)
    fillState(prev_dir, -0.02, 0.004);
#endif

    Vector<Real>              prm;
    algebra::DenseSystemMatrix next_jac;
    eq.assembleNextStateJac(
        0, x_next, x, prm, next_jac);
#if defined(FEMX_HAS_ENZYME)
    algebra::DenseSystemMatrix prev_jac;
    eq.assemblePrevStateJac(
        0, x_next, x, prm, prev_jac);
#endif

    Vector<Real> next_prod;
    next_jac.apply(next_dir, next_prod);
#if defined(FEMX_HAS_ENZYME)
    Vector<Real> prev_prod;
    prev_jac.apply(prev_dir, prev_prod);
#endif

    const Real epsilon = 1.0e-7;

    Vector<Real> x_next_pert = x_next;
    for (Index i = 0; i < x_next_pert.size(); ++i)
    {
      x_next_pert[i] += epsilon * next_dir[i];
    }

    Vector<Real> res;
    Vector<Real> res_pert;
    eq.res(0, x_next, x, prm, res);
    eq.res(0, x_next_pert, x, prm, res_pert);

    for (Index i = 0; i < res.size(); ++i)
    {
      const Real fd  = (res_pert[i] - res[i]) / epsilon;
      status        *= isClose(fd, next_prod[i], 1.0e-7);
    }

#if defined(FEMX_HAS_ENZYME)
    Vector<Real> x_pert = x;
    for (Index i = 0; i < x_pert.size(); ++i)
    {
      x_pert[i] += epsilon * prev_dir[i];
    }

    eq.res(0, x_next, x_pert, prm, res_pert);

    for (Index i = 0; i < res.size(); ++i)
    {
      const Real fd  = (res_pert[i] - res[i]) / epsilon;
      status        *= isClose(fd, prev_prod[i], 1.0e-7);
    }
#endif

    return status.report(__func__);
  }

  TestOutcome paramJacobianHasZeroColumns()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
    LagrangeQuadQ1 element;
    MixedFESpace   space = makeSpace(mesh, element);

    TimeNavierStokesParameters ns_prm;
    NavierStokesEquation       eq(space, ns_prm);

    Vector<Real> x_next(eq.numStates());
    Vector<Real> x(eq.numStates());
    Vector<Real> prm;

    algebra::DenseSystemMatrix param_jac;
    eq.assembleParamJac(0, x_next, x, prm, param_jac);

    status *= (eq.numParams() == 0);
    status *= (param_jac.numRows() == eq.numRes());
    status *= (param_jac.numCols() == 0);

    return status.report(__func__);
  }

  TestOutcome velocityBoundaryControlUsesVelocityDofs()
  {
    TestStatus status;
    status = true;

    Mesh mesh = Mesh::makeStructuredQuad(1, 1);
    addBoundaryFacet(mesh, 7, "control", {1, 3});

    LagrangeQuadQ1 element;
    MixedFESpace   space = makeSpace(mesh, element);
    const auto     u_dof = space.field(0);

    DirichletControl control =
        makeVelocityControl(space, "control");

    status *= (control.numDofs() == 4);
    status *= (control.stateDof(0) == u_dof.globalDof(1, 0));
    status *= (control.stateDof(1) == u_dof.globalDof(1, 1));
    status *= (control.stateDof(2) == u_dof.globalDof(3, 0));
    status *= (control.stateDof(3) == u_dof.globalDof(3, 1));
    status *= (control.paramIndex(1, 2) == 6);

    DirichletControl by_tag =
        makeVelocityControl(space, 7);
    status *= (by_tag.numDofs() == control.numDofs());
    for (Index i = 0; i < control.numDofs(); ++i)
    {
      status *= (by_tag.stateDof(i) == control.stateDof(i));
    }

    return status.report(__func__);
  }

  TestOutcome dirichletControlEquationUsesControlParameters()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
    LagrangeQuadQ1 element;
    MixedFESpace   space = makeSpace(mesh, element);
    const auto     u_dof = space.field(0);
    const auto     p_dof = space.field(1);

    TimeNavierStokesParameters ns_prm;
    ns_prm.steps     = 2;
    ns_prm.dt        = 0.25;
    ns_prm.fluid.rho = 1.4;
    ns_prm.fluid.mu  = 0.03;

    NavierStokesEquation base_eq(space, ns_prm);
    DirichletControl     control(
        {u_dof.globalDof(1, 0), u_dof.globalDof(3, 1)});
    Vector<Index> fixed_dofs = {p_dof.globalDof(0, 0)};
    Vector<Real>  fixed_values = {0.75, -1.25};
    problem::TimeDirichletControlEquation eq(
        base_eq, control, fixed_dofs, 0, -1, fixed_values);

    Vector<Real> x_next(eq.numStates());
    Vector<Real> x(eq.numStates());
    fillState(x_next, 0.11, 0.013);
    fillState(x, -0.07, 0.017);

    Vector<Real> prm(eq.numParams());
    prm[0] = 0.20;
    prm[1] = -0.30;
    prm[2] = 0.40;
    prm[3] = -0.50;

    Vector<Real> res;
    eq.res(1, x_next, x, prm, res);

    const Index row0       = control.stateDof(0);
    const Index row1       = control.stateDof(1);
    const Index fixed_row  = fixed_dofs[0];
    status                *= (eq.numParams() == 4);
    status                *= isEqual(res[row0], x_next[row0] - prm[2]);
    status                *= isEqual(res[row1], x_next[row1] - prm[3]);
    status                *= isEqual(res[fixed_row],
                                     x_next[fixed_row] - fixed_values[1]);

    algebra::DenseSystemMatrix param_jac;
#if defined(FEMX_HAS_ENZYME)
    algebra::DenseSystemMatrix next_jac;
    eq.assembleNextStateJac(1, x_next, x, prm, next_jac);
    algebra::DenseSystemMatrix prev_jac;
    eq.assemblePrevStateJac(1, x_next, x, prm, prev_jac);
#endif
    eq.assembleParamJac(1, x_next, x, prm, param_jac);

#if defined(FEMX_HAS_ENZYME)
    for (Index col = 0; col < eq.numStates(); ++col)
    {
      status *= isEqual(next_jac.matrix()(row0, col),
                        col == row0 ? 1.0 : 0.0);
      status *= isEqual(next_jac.matrix()(row1, col),
                        col == row1 ? 1.0 : 0.0);
      status *= isEqual(prev_jac.matrix()(row0, col), 0.0);
      status *= isEqual(prev_jac.matrix()(row1, col), 0.0);
      status *= isEqual(next_jac.matrix()(fixed_row, col),
                        col == fixed_row ? 1.0 : 0.0);
      status *= isEqual(prev_jac.matrix()(fixed_row, col), 0.0);
    }
#endif

    for (Index col = 0; col < eq.numParams(); ++col)
    {
      status *= isEqual(param_jac.matrix()(row0, col),
                        col == 2 ? -1.0 : 0.0);
      status *= isEqual(param_jac.matrix()(row1, col),
                        col == 3 ? -1.0 : 0.0);
      status *= isEqual(param_jac.matrix()(fixed_row, col), 0.0);
    }

    Vector<Real> param_dir(eq.numParams());
    fillState(param_dir, -0.09, 0.021);
    Vector<Real> param_apply_from_matrix;
    Vector<Real> param_apply_direct;
    param_jac.apply(param_dir, param_apply_from_matrix);
    eq.applyParamJac(
        1, x_next, x, prm, param_dir, param_apply_direct);
    for (Index row = 0; row < eq.numRes(); ++row)
    {
      status *= isEqual(param_apply_direct[row],
                        param_apply_from_matrix[row]);
    }

    Vector<Real> lambda(eq.numRes());
    fillState(lambda, 0.13, -0.007);
    Vector<Real> param_transpose_from_matrix;
    Vector<Real> param_transpose_direct;
    param_jac.applyT(lambda, param_transpose_from_matrix);
    eq.applyParamJacT(
        1, x_next, x, prm, lambda, param_transpose_direct);
    for (Index col = 0; col < eq.numParams(); ++col)
    {
      status *= isEqual(param_transpose_direct[col],
                        param_transpose_from_matrix[col]);
    }

#if defined(FEMX_HAS_ENZYME)
    const CsrPattern           pattern = assembly::SparsityPatternBuilder::build(space);
    algebra::SparseSystemMatrix sparse_next(pattern);
    algebra::SparseSystemMatrix sparse_prev(pattern);
    eq.assembleNextStateJac(1, x_next, x, prm, sparse_next);
    eq.assemblePrevStateJac(1, x_next, x, prm, sparse_prev);

    Vector<Real> dir(eq.numStates());
    fillState(dir, 0.07, 0.011);
    Vector<Real> next_prod;
    Vector<Real> prev_prod;
    sparse_next.apply(dir, next_prod);
    sparse_prev.apply(dir, prev_prod);

    status *= isEqual(next_prod[row0], dir[row0]);
    status *= isEqual(next_prod[row1], dir[row1]);
    status *= isEqual(next_prod[fixed_row], dir[fixed_row]);
    status *= isEqual(prev_prod[row0], 0.0);
    status *= isEqual(prev_prod[row1], 0.0);
    status *= isEqual(prev_prod[fixed_row], 0.0);
#endif

    return status.report(__func__);
  }

  TestOutcome dirichletControlEquationSupportsParameterOffset()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
    LagrangeQuadQ1 element;
    MixedFESpace   space = makeSpace(mesh, element);
    const auto     u_dof = space.field(0);

    TimeNavierStokesParameters ns_prm;
    ns_prm.steps     = 2;
    ns_prm.dt        = 0.25;
    ns_prm.fluid.rho = 1.4;
    ns_prm.fluid.mu  = 0.03;

    NavierStokesEquation base_eq(space, ns_prm);
    DirichletControl     control(
        {u_dof.globalDof(1, 0), u_dof.globalDof(3, 1)});
    const Index              offset    = 3;
    const Index              total_prm = offset + control.numParams(ns_prm.steps) + 2;
    problem::TimeDirichletControlEquation eq(
        base_eq, control, {}, offset, total_prm);

    Vector<Real> x_next(eq.numStates());
    Vector<Real> x(eq.numStates());
    fillState(x_next, 0.11, 0.013);
    fillState(x, -0.07, 0.017);

    Vector<Real> full_prm(eq.numParams());
    fillState(full_prm, 0.20, -0.03);

    Vector<Real> res;
    eq.res(1, x_next, x, full_prm, res);

    const Index row0  = control.stateDof(0);
    const Index row1  = control.stateDof(1);
    const Index col0  = offset + control.paramIndex(1, 0);
    const Index col1  = offset + control.paramIndex(1, 1);
    status           *= (eq.numParams() == total_prm);
    status           *= isEqual(res[row0], x_next[row0] - full_prm[col0]);
    status           *= isEqual(res[row1], x_next[row1] - full_prm[col1]);

    algebra::DenseSystemMatrix param_jac;
    eq.assembleParamJac(1, x_next, x, full_prm, param_jac);
    for (Index col = 0; col < eq.numParams(); ++col)
    {
      status *= isEqual(param_jac.matrix()(row0, col),
                        col == col0 ? -1.0 : 0.0);
      status *= isEqual(param_jac.matrix()(row1, col),
                        col == col1 ? -1.0 : 0.0);
    }

    Vector<Real> lambda(eq.numRes());
    fillState(lambda, 0.13, -0.007);
    Vector<Real> param_transpose;
    eq.applyParamJacT(
        1, x_next, x, full_prm, lambda, param_transpose);
    for (Index col = 0; col < eq.numParams(); ++col)
    {
      const Real expected =
          col == col0 ? -lambda[row0]
                      : (col == col1 ? -lambda[row1] : 0.0);
      status *= isEqual(param_transpose[col], expected);
    }

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  std::cout << "Running time Navier-Stokes residual equation tests:\n";

  femx::tests::NavierStokesEquationTests test;

  femx::tests::TestingResults result;
  result += test.residualAndJacobiansHaveConsistentSizes();
  result += test.jacobiansMatchFiniteDifferences();
  result += test.paramJacobianHasZeroColumns();
  result += test.velocityBoundaryControlUsesVelocityDofs();
  result += test.dirichletControlEquationUsesControlParameters();
  result += test.dirichletControlEquationSupportsParameterOffset();

  return result.summary();
}
