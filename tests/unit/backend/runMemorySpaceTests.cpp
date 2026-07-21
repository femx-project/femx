#include <array>
#include <string>
#include <type_traits>
#include <utility>

#include "TestHelper.hpp"
#include <femx/common/Context.hpp>
#include <femx/linalg/Backend.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace tests
{
namespace
{

template <class T, class = void>
struct HasContextlessSetZero : std::false_type
{
};

template <class T>
struct HasContextlessSetZero<
    T,
    std::void_t<decltype(std::declval<T&>().setZero())>> : std::true_type
{
};

template <class T, class = void>
struct HasContextSetZero : std::false_type
{
};

template <class T>
struct HasContextSetZero<
    T,
    std::void_t<decltype(std::declval<T&>().setZero(
        std::declval<CudaContext&>()))>> : std::true_type
{
};

template <class T, class = void>
struct HasContextlessResizeOrZero : std::false_type
{
};

template <class T>
struct HasContextlessResizeOrZero<
    T,
    std::void_t<decltype(resizeOrZero(std::declval<T&>(), Index{}))>>
  : std::true_type
{
};

template <class T, class = void>
struct HasContextResizeOrZero : std::false_type
{
};

template <class T>
struct HasContextResizeOrZero<
    T,
    std::void_t<decltype(resizeOrZero(std::declval<T&>(),
                                      Index{},
                                      std::declval<CudaContext&>()))>>
  : std::true_type
{
};

TestOutcome hostAndDeviceTypesAreDistinct()
{
  TestStatus status(__func__);

  status *= !std::is_same<HostVector, DeviceVector>::value;
  status *= !std::is_same<HostIndexVector, DeviceIndexVector>::value;
  status *= std::is_same<typename std::remove_pointer<
                             decltype(HostVector{}.data())>::type,
                         Real>::value;

  return status.report();
}

TestOutcome executionIsSeparateFromStorage()
{
  TestStatus status(__func__);

  using linalg::CudaCsrBackend;
  using linalg::HostCsrBackend;

  status *= linalg::is_backend_v<HostCsrBackend>;
  status *= linalg::is_backend_v<CudaCsrBackend>;
  status *= HostCsrBackend::space == MemorySpace::Host;
  status *= CudaCsrBackend::space == MemorySpace::Device;
  status *= std::is_same<HostCsrBackend::Mat, HostCsrMatrix>::value;
  status *= std::is_same<CudaCsrBackend::Mat, DeviceCsrMatrix>::value;
  status *= std::is_same<HostCsrBackend::Ctx, CpuContext>::value;
  status *= std::is_same<CudaCsrBackend::Ctx, CudaContext>::value;

  return status.report();
}

TestOutcome deviceStorageRequiresExplicitStreamSemantics()
{
  TestStatus status(__func__);

  status *= !std::is_copy_constructible<DeviceVector>::value;
  status *= !std::is_copy_assignable<DeviceVector>::value;
  status *= std::is_nothrow_move_constructible<DeviceVector>::value;
  status *= std::is_nothrow_move_assignable<DeviceVector>::value;
  status *= !HasContextlessSetZero<HostVector>::value;
  status *= !HasContextlessSetZero<DeviceVector>::value;
  status *= !HasContextSetZero<DeviceVector>::value;
  status *= !HasContextlessResizeOrZero<HostVector>::value;
  status *= !HasContextlessResizeOrZero<DeviceVector>::value;
  status *= !HasContextResizeOrZero<DeviceVector>::value;
  status *= std::is_constructible<linalg::HostVectorHandler,
                                  CpuContext&>::value;
  status *= std::is_constructible<linalg::CudaVectorHandler,
                                  CudaContext&>::value;
  status *= HasContextlessSetZero<state::TimeTrajectory>::value;
  status *= !HasContextSetZero<state::TimeTrajectory>::value;
  status *= std::is_copy_constructible<state::TimeTrajectory>::value;
  status *= std::is_same<
      decltype(std::declval<state::TimeTrajectory&>().level(Index{})),
      HostVectorView>::value;
  status *= std::is_same<
      decltype(std::declval<const state::TimeTrajectory&>().level(Index{})),
      HostConstVectorView>::value;

  return status.report();
}

TestOutcome hostVectorOwnsOnlyHostValues()
{
  TestStatus status(__func__);

  HostVector vals{1.0, 2.0, 3.0};
  HostVector copy = vals;
  copy[1]         = 7.0;

  status *= vals.size() == 3;
  status *= vals[1] == 2.0;
  status *= copy[1] == 7.0;

  HostVectorView view(vals.data(), vals.size());
  view[0]  = 4.0;
  status  *= vals[0] == 4.0;

  HostVector from_view(view);
  status *= from_view.data() != vals.data();
  status *= from_view[0] == 4.0;
  status *= from_view[2] == 3.0;

  return status.report();
}

TestOutcome arrayUsesHostVectorStorage()
{
  TestStatus status(__func__);

  Array<const char*> labels{"state", "residual", "jacobian"};
  status *= std::is_same<Array<Index>, HostIndexVector>::value;
  status *= labels.size() == 3;
  status *= labels[1] == std::string("residual");

  HostIndexVector indices{2, 4, 6};
  status *= indices.size() == 3;
  status *= indices[2] == 6;

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main()
{
  femx::tests::TestingResults results;
  results += femx::tests::hostAndDeviceTypesAreDistinct();
  results += femx::tests::executionIsSeparateFromStorage();
  results += femx::tests::deviceStorageRequiresExplicitStreamSemantics();
  results += femx::tests::hostVectorOwnsOnlyHostValues();
  results += femx::tests::arrayUsesHostVectorStorage();
  return results.summary();
}
