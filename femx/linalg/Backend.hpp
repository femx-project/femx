#pragma once

#include <type_traits>

#include <femx/common/Types.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::linalg
{

/** @brief Provide serial CPU execution over Host CSR storage. */
struct HostCsrBackend
{
  static constexpr MemorySpace space = MemorySpace::Host; ///< Storage memory space.

  using Vec       = HostVector<Real>;
  using VecView   = HostVectorView<Real>;
  using ConstView = HostVectorView<const Real>;
  using Mat       = HostCsrMatrix;
  using Pattern   = HostCsrPattern;
};

/** @brief Provide CUDA execution over Device CSR storage. */
struct CudaCsrBackend
{
  static constexpr MemorySpace space = MemorySpace::Device; ///< Storage memory space.

  using Vec       = DeviceVector<Real>;
  using VecView   = DeviceVectorView<Real>;
  using ConstView = DeviceVectorView<const Real>;
  using Mat       = DeviceCsrMatrix;
  using Pattern   = DeviceCsrPattern;
};

/** @brief Map a memory space to its native CSR backend. */
template <MemorySpace Space>
struct CsrBackendFor;

/** @brief Map Host memory to the Host CSR backend. */
template <>
struct CsrBackendFor<MemorySpace::Host>
{
  using Type = HostCsrBackend;
};

/** @brief Map Device memory to the CUDA CSR backend. */
template <>
struct CsrBackendFor<MemorySpace::Device>
{
  using Type = CudaCsrBackend;
};

/** @brief Select the native CSR backend for a memory space. */
template <MemorySpace Space>
using CsrBackend = typename CsrBackendFor<Space>::Type;

/** @brief Detect the minimal type contract required from a femx backend. */
template <class Backend, class = void>
struct IsBackend : std::false_type
{
};

/** @brief Recognize a type that satisfies the femx backend contract. */
template <class Backend>
struct IsBackend<Backend,
                 std::void_t<typename Backend::Vec,
                             typename Backend::VecView,
                             typename Backend::ConstView,
                             typename Backend::Mat,
                             typename Backend::Pattern,
                             decltype(Backend::space)>> : std::true_type
{
};

/** @brief Report whether a type satisfies the femx backend contract. */
template <class Backend>
inline constexpr bool is_backend_v = IsBackend<Backend>::value;

static_assert(is_backend_v<HostCsrBackend>,
              "HostCsrBackend does not satisfy the backend contract");
static_assert(is_backend_v<CudaCsrBackend>,
              "CudaCsrBackend does not satisfy the backend contract");

} // namespace femx::linalg
