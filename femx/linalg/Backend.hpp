#pragma once

#include <type_traits>

#include <femx/common/Context.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::linalg
{

/** @brief Serial CPU execution over Host CSR storage. */
struct HostCsrBackend
{
  static constexpr MemorySpace space = MemorySpace::Host;

  using Vec       = HostVector;
  using VecView   = HostVectorView;
  using ConstView = HostConstVectorView;
  using Mat       = HostCsrMatrix;
  using Graph     = HostCsrGraph;
  using Ctx       = CpuContext;
};

/** @brief CUDA execution over Device CSR storage. */
struct CudaCsrBackend
{
  static constexpr MemorySpace space = MemorySpace::Device;

  using Vec       = DeviceVector;
  using VecView   = DeviceVectorView;
  using ConstView = DeviceConstVectorView;
  using Mat       = DeviceCsrMatrix;
  using Graph     = DeviceCsrGraph;
  using Ctx       = CudaContext;
};

/** @brief Detect the minimal type contract required from a femx backend. */
template <class Backend, class = void>
struct IsBackend : std::false_type
{
};

template <class Backend>
struct IsBackend<Backend,
                 std::void_t<typename Backend::Vec,
                             typename Backend::VecView,
                             typename Backend::ConstView,
                             typename Backend::Mat,
                             typename Backend::Graph,
                             typename Backend::Ctx,
                             decltype(Backend::space)>> : std::true_type
{
};

template <class Backend>
inline constexpr bool is_backend_v = IsBackend<Backend>::value;

static_assert(is_backend_v<HostCsrBackend>,
              "HostCsrBackend does not satisfy the backend contract");
static_assert(is_backend_v<CudaCsrBackend>,
              "CudaCsrBackend does not satisfy the backend contract");

} // namespace femx::linalg
