#include <cmath>
#include <iostream>

#include <femx/assembly/EnzymeTimeVolumeKernel.hpp>
#include <femx/assembly/TimeFEMResidual.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/linalg/native/DenseAssemblyMatrix.hpp>
#include <tests/TestBase.hpp>

namespace femx
{
namespace tests
{

namespace
{

void localTimeVolumeResidual(Index       step,
                             Index       elem,
                             Index       num_qpts,
                             Index       num_nodes,
                             Index       dim,
                             Index       num_residuals,
                             Index       num_prev_states,
                             Index       num_next_states,
                             Index       num_params,
                             const Real* N,
                             const Real* dNdx,
                             const Real* JxW,
                             const Real* prev,
                             const Real* nxt,
                             const Real* prm,
                             Real*       out)
{
  (void) elem;
  (void) dim;
  (void) num_prev_states;
  (void) num_next_states;
  (void) num_params;
  (void) dNdx;

  for (Index i = 0; i < num_residuals; ++i)
  {
    out[i] = 0.0;
  }

  for (Index iq = 0; iq < num_qpts; ++iq)
  {
    const Real* Nq = N + iq * num_nodes;

    Real previous_q = 0.0;
    Real next_q     = 0.0;
    Real prm_q      = 0.0;
    for (Index a = 0; a < num_nodes; ++a)
    {
      previous_q += Nq[a] * prev[a];
      next_q     += Nq[a] * nxt[a];
      prm_q      += Nq[a] * prm[a];
    }

    const Real source = static_cast<Real>(step + 1) * previous_q
                        + next_q * next_q + 2.0 * prm_q;
    for (Index a = 0; a < num_nodes; ++a)
    {
      out[a] += Nq[a] * source * JxW[iq];
    }
  }
}

bool near(Real actual, Real exp)
{
  return std::abs(actual - exp) < 1.0e-10;
}

void fill(Vector<Real>& vals, std::initializer_list<Real> input)
{
  vals = input;
}

} // namespace

class EnzymeTimeVolumeKernelTests : public TestBase
{
public:
  TestOutcome assemblesTimeJacobians()
  {
    TestStatus status;
    status = true;

    Mesh           mesh = Mesh::makeStructuredQuad(1, 1);
    LagrangeQuadQ1 elem;
    FESpace        space(&mesh, &elem);
    space.setup();

    const auto                                                quad = GaussQuadrature::quadrilateral(1);
    assembly::EnzymeTimeVolumeKernel<localTimeVolumeResidual> ker(
        space, quad, 4, 4, 4, 4);

    assembly::TimeFEMResidual res(2,
                                  DofLayout(space),
                                  DofLayout(space),
                                  DofLayout(space),
                                  DofLayout(space),
                                  ker);

    Vector<Real> prev(4);
    Vector<Real> next(4);
    Vector<Real> prm(4);
    fill(prev, {1.0, 2.0, -1.0, 0.5});
    fill(next, {0.25, -0.5, 1.0, 0.75});
    fill(prm, {0.1, 0.2, -0.3, 0.4});

    problem::TimeContext ctx;
    ctx.step = 1;
    ctx.prev = &prev;
    ctx.nxt  = &next;
    ctx.prm  = &prm;

    Vector<Real> out;
    res.res(ctx, out);

    const Real previous_q = 0.25 * (prev[0] + prev[1] + prev[2] + prev[3]);
    const Real next_q =
        0.25 * (next[0] + next[1] + next[2] + next[3]);
    const Real prm_q = 0.25 * (prm[0] + prm[1] + prm[2] + prm[3]);
    const Real source =
        2.0 * previous_q + next_q * next_q + 2.0 * prm_q;

    status *= (out.size() == 4);
    for (Index i = 0; i < out.size(); ++i)
    {
      status *= near(out[i], 0.25 * source);
    }

    linalg::DenseAssemblyMatrix J_prev;
    linalg::DenseAssemblyMatrix J_next;
    linalg::DenseAssemblyMatrix J_param;
    status *= res.assembleJac(
        ctx, problem::VariableBlock::PrevState, J_prev);
    status *= res.assembleJac(
        ctx, problem::VariableBlock::NextState, J_next);
    status *= res.assembleJac(
        ctx, problem::VariableBlock::Param, J_param);

    const Real expected_previous = 0.25 * 2.0 * 0.25;
    const Real expected_next     = 0.25 * 2.0 * next_q * 0.25;
    const Real expected_param    = 0.25 * 2.0 * 0.25;

    for (Index i = 0; i < 4; ++i)
    {
      for (Index j = 0; j < 4; ++j)
      {
        status *= near(J_prev.mat()(i, j),
                       expected_previous);
        status *= near(J_next.mat()(i, j), expected_next);
        status *= near(J_param.mat()(i, j), expected_param);
      }
    }

    return status.report(__func__);
  }
};

} // namespace tests
} // namespace femx

int main(int, char**)
{
  femx::tests::TestingResults              results;
  femx::tests::EnzymeTimeVolumeKernelTests test;

  results += test.assemblesTimeJacobians();

  return results.summary();
}
