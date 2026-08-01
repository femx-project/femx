#include <cmath>
#include <memory>

#include "TestHelper.hpp"
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/fem/TimePointInterpolator.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/host/HostContext.hpp>

namespace femx
{
namespace tests
{
namespace
{

using fem::CudaTimePointInterpolator;
using fem::TimePointInterpolator;

class InterpolatorFixture
{
public:
  InterpolatorFixture()
    : mesh(fem::Mesh::makeStructuredQuad(1, 1)),
      field(&mesh, &elem, 2)
  {
    space.addField(field);
    space.setup();
    op = std::make_unique<TimePointInterpolator>(
        2,
        space,
        0,
        HostVector<Point3>{{0.25, 0.25, 0.0}, {0.75, 0.5, 0.0}},
        HostVector<Index>{0, 1},
        0);
  }

  fem::Mesh                              mesh;
  fem::LagrangeQuadQ1                    elem;
  fem::FESpace                           field;
  fem::MixedFESpace                      space;
  std::unique_ptr<TimePointInterpolator> op;
};

bool near(Real lhs, Real rhs, Real tol = 1.0e-12)
{
  return std::abs(lhs - rhs) <= tol;
}

bool near(const HostVector<Real>& lhs,
          const HostVector<Real>& rhs,
          Real                    tol = 1.0e-12)
{
  if (lhs.size() != rhs.size())
  {
    return false;
  }
  for (Index i = 0; i < lhs.size(); ++i)
  {
    if (!near(lhs[i], rhs[i], tol))
    {
      return false;
    }
  }
  return true;
}

Real innerProduct(const HostVector<Real>& lhs, const HostVector<Real>& rhs)
{
  Real val = 0.0;
  for (Index i = 0; i < lhs.size(); ++i)
  {
    val += lhs[i] * rhs[i];
  }
  return val;
}

TestOutcome hostFlatObserveAndTranspose()
{
  TestStatus             status(__func__);
  InterpolatorFixture    fixture;
  const auto&            op = *fixture.op;
  const HostVector<Real> state{1.0, 10.0, 3.0, 20.0, 5.0, 30.0, 7.0, 40.0};
  const HostVector<Real> dir{1.25, -0.5, 2.0, 0.75};
  const HostVector<Real> prm;

  HostVector<Real> expected_obs;
  op.observe(1, state, prm, expected_obs);

  HostVector<Real>    flat_obs(op.numObservations());
  linalg::HostContext ctx;
  auto&               mat_handler = ctx.matrixHandler();
  mat_handler.matvec(op.data().matrix(), state.view(), flat_obs.view());

  HostVector<Real> expected_trans;
  op.applyStateJacT(1, state, prm, dir, expected_trans);

  HostVector<Real> flat_trans(op.numStates());
  mat_handler.matvecT(
      op.data().matrix(), dir.view(), flat_trans.view(), 1.0, 1.0);

  status *= op.data().numObservations() == 4;
  status *= op.data().numEntries() == 16;
  status *= near(flat_obs, HostVector<Real>{2.5, 17.5, 4.5, 27.5});
  status *= near(flat_obs, expected_obs);
  status *= near(flat_trans, expected_trans);
  status *= near(innerProduct(flat_obs, dir), innerProduct(state, flat_trans));
  return status.report();
}

#if defined(FEMX_HAS_CUDA)
TestOutcome cudaObserveAndTransposeMatchHost()
{
  TestStatus status(__func__);
  if (!linalg::CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  InterpolatorFixture    fixture;
  const auto&            op = *fixture.op;
  const HostVector<Real> state{1.0, 10.0, 3.0, 20.0, 5.0, 30.0, 7.0, 40.0};
  const HostVector<Real> dir{1.25, -0.5, 2.0, 0.75};
  const HostVector<Real> prm;

  HostVector<Real> expected_obs;
  HostVector<Real> expected_trans;
  op.observe(0, state, prm, expected_obs);
  op.applyStateJacT(0, state, prm, dir, expected_trans);

  linalg::CudaContext       ctx;
  auto&                     vec_handler = ctx.vectorHandler();
  CudaTimePointInterpolator d_op;
  DeviceVector<Real>        d_state;
  DeviceVector<Real>        d_dir;
  DeviceVector<Real>        d_obs(op.numObservations());
  DeviceVector<Real>        d_trans(op.numStates());

  fem::copy(op, d_op, ctx);
  vec_handler.copy(state, d_state);
  vec_handler.copy(dir, d_dir);

  const inverse::DeviceTimeObservationOperator& iface     = d_op;
  const Real*                                   obs_ptr   = d_obs.data();
  const Real*                                   trans_ptr = d_trans.data();
  iface.observe(1, d_state.view(), d_obs.view(), ctx);
  vec_handler.zero(d_trans.view());
  iface.addStateJacT(1, d_dir.view(), d_trans.view(), ctx);

  HostVector<Real> got_obs;
  HostVector<Real> got_trans;
  vec_handler.copy(d_obs, got_obs);
  vec_handler.copy(d_trans, got_trans);
  ctx.sync();

  status *= near(got_obs, expected_obs);
  status *= near(got_trans, expected_trans);
  status *= near(innerProduct(got_obs, dir), innerProduct(state, got_trans));
  status *= iface.numSteps() == op.numSteps();
  status *= iface.numStates() == op.numStates();
  status *= iface.numObservations() == op.numObservations();
  status *= d_obs.data() == obs_ptr;
  status *= d_trans.data() == trans_ptr;
  return status.report();
}
#endif

} // namespace
} // namespace tests
} // namespace femx

int main()
{
  femx::tests::TestingResults results;
  results += femx::tests::hostFlatObserveAndTranspose();
#if defined(FEMX_HAS_CUDA)
  results += femx::tests::cudaObserveAndTransposeMatchHost();
#endif
  return results.summary();
}
