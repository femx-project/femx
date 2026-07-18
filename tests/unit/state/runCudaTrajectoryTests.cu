#include <cuda_runtime_api.h>

#include <exception>
#include <iostream>
#include <type_traits>

#include "TestHelper.hpp"
#include <femx/common/Context.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace tests
{
namespace
{

__global__ void fillTrajectory(
    state::TrajectoryView<MemorySpace::Device, Real> tr)
{
  const Index i = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= tr.size())
  {
    return;
  }

  const Index level = i / tr.numStates();
  const Index state = i - level * tr.numStates();
  tr[level][state]  = 10.0 * static_cast<Real>(level)
                     + static_cast<Real>(state + 1);
}

TestOutcome deviceTrajectoryUsesViewsAndExplicitWholeCopy()
{
  TestStatus status(__func__);
  if (!CudaContext::available())
  {
    status.skipTest();
    return status.report();
  }

  try
  {
    static_assert(
        std::is_trivially_copyable<
            state::TrajectoryView<MemorySpace::Device, Real>>::value,
        "TrajectoryView must be passed to CUDA kernels by value");

    CudaContext                 ctx;
    state::DeviceTimeTrajectory tr(2, 3);
    Real* const                 storage = tr.data();

    tr.resize(2, 3);
    status *= tr.data() == storage;

    constexpr int threads = 32;
    fillTrajectory<<<1, threads, 0, static_cast<cudaStream_t>(ctx.stream())>>>(
        tr.view());
    device::checkLastError();

    state::TimeTrajectory host;
    state::copy(tr, host, ctx);
    ctx.synchronize();

    status *= host.numSteps() == 2;
    status *= host.numTimeLevels() == 3;
    status *= host.numStates() == 3;
    status *= host.level(0)[0] == 1.0;
    status *= host.level(1)[2] == 13.0;
    status *= host.level(2)[1] == 22.0;

    host.level(1)[0] = -4.0;
    state::copy(host, tr, ctx);
    status *= tr.data() == storage;

    state::TimeTrajectory roundtrip;
    state::copy(tr, roundtrip, ctx);
    ctx.synchronize();
    status *= roundtrip.level(1)[0] == -4.0;

    tr.setZero(ctx);
    state::copy(tr, roundtrip, ctx);
    ctx.synchronize();
    for (Index i = 0; i < roundtrip.size(); ++i)
    {
      status *= roundtrip.data()[i] == 0.0;
    }
  }
  catch (const std::exception& error)
  {
    std::cout << "    exception: " << error.what() << '\n';
    status *= false;
  }

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main()
{
  femx::tests::TestingResults results;
  results += femx::tests::deviceTrajectoryUsesViewsAndExplicitWholeCopy();
  return results.summary();
}
