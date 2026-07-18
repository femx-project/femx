#include <cmath>
#include <memory>

#include "TestHelper.hpp"
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/fem/TimePointInterpolator.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>

namespace femx
{
namespace tests
{
namespace
{

using fem::DeviceTimePointInterpolator;
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
        Array<Point3>{{0.25, 0.25, 0.0}, {0.75, 0.5, 0.0}},
        Array<Index>{0, 1},
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

bool near(const HostVector& lhs,
          const HostVector& rhs,
          Real              tol = 1.0e-12)
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

Real innerProduct(const HostVector& lhs, const HostVector& rhs)
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
  TestStatus          status(__func__);
  InterpolatorFixture fixture;
  const auto&         op = *fixture.op;
  const HostVector    state{1.0, 10.0, 3.0, 20.0, 5.0, 30.0, 7.0, 40.0};
  const HostVector    dir{1.25, -0.5, 2.0, 0.75};
  const HostVector    prm;

  HostVector expected_obs;
  op.observe(1, state, prm, expected_obs);

  HostVector flat_obs(op.numObservations());
  fem::observe(op.data().view(), state.view(), flat_obs.view());

  HostVector expected_tr;
  op.applyStateJacT(1, state, prm, dir, expected_tr);

  HostVector flat_tr(op.numStates());
  flat_tr.setZero();
  fem::addStateJacT(op.data().view(), dir.view(), flat_tr.view());

  status *= op.data().numObservations() == 4;
  status *= op.data().numEntries() == 16;
  status *= near(flat_obs, HostVector{2.5, 17.5, 4.5, 27.5});
  status *= near(flat_obs, expected_obs);
  status *= near(flat_tr, expected_tr);
  status *= near(innerProduct(flat_obs, dir), innerProduct(state, flat_tr));
  return status.report();
}

#if defined(FEMX_HAS_CUDA)
TestOutcome cudaObserveAndTransposeMatchHost()
{
  TestStatus status(__func__);
  if (!CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  InterpolatorFixture fixture;
  const auto&         op = *fixture.op;
  const HostVector    state{1.0, 10.0, 3.0, 20.0, 5.0, 30.0, 7.0, 40.0};
  const HostVector    dir{1.25, -0.5, 2.0, 0.75};
  const HostVector    prm;

  HostVector expected_obs;
  HostVector expected_tr;
  op.observe(0, state, prm, expected_obs);
  op.applyStateJacT(0, state, prm, dir, expected_tr);

  CudaContext                 ctx;
  DeviceTimePointInterpolator dev_op;
  DeviceVector                dev_state;
  DeviceVector                dev_dir;
  DeviceVector                dev_obs(op.numObservations());
  DeviceVector                dev_tr(op.numStates());

  fem::copy(op, dev_op, ctx);
  femx::copy(state, dev_state, ctx);
  femx::copy(dir, dev_dir, ctx);

  const inverse::DeviceTimeObservationOperator& iface   = dev_op;
  const Real*                                   obs_ptr = dev_obs.data();
  const Real*                                   tr_ptr  = dev_tr.data();
  iface.observe(1, dev_state.view(), dev_obs.view(), ctx);
  dev_tr.setZero(ctx);
  iface.addStateJacT(1, dev_dir.view(), dev_tr.view(), ctx);

  HostVector got_obs;
  HostVector got_tr;
  femx::copy(dev_obs, got_obs, ctx);
  femx::copy(dev_tr, got_tr, ctx);
  ctx.synchronize();

  status *= near(got_obs, expected_obs);
  status *= near(got_tr, expected_tr);
  status *= near(innerProduct(got_obs, dir), innerProduct(state, got_tr));
  status *= iface.numSteps() == op.numSteps();
  status *= iface.numStates() == op.numStates();
  status *= iface.numObservations() == op.numObservations();
  status *= dev_obs.data() == obs_ptr;
  status *= dev_tr.data() == tr_ptr;
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
