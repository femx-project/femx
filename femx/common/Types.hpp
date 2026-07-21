#pragma once

#include <cstdint>
#include <limits>

namespace femx
{

/** @brief Storage location used by backend-aware containers and views. */
enum class MemorySpace
{
  Host,  ///< CPU-addressable memory.
  Device ///< CUDA device memory.
};

/** @brief Scalar type used by finite-element and linear-algebra operations. */
using Real  = double;
/** @brief Signed index type used by meshes, DOFs, and sparse matrices. */
using Index = std::int32_t;

template <MemorySpace Space, class T = Real>
class Vector;

template <class T>
using Array = Vector<MemorySpace::Host, T>;

template <MemorySpace Space, class T>
class VectorView;

template <MemorySpace Space>
class CsrPattern;

template <MemorySpace Space>
class CsrMatrix;

using HostVector   = Vector<MemorySpace::Host>;
using DeviceVector = Vector<MemorySpace::Device>;

using HostIndexVector   = Vector<MemorySpace::Host, Index>;
using DeviceIndexVector = Vector<MemorySpace::Device, Index>;

using HostCsrPattern   = CsrPattern<MemorySpace::Host>;
using DeviceCsrPattern = CsrPattern<MemorySpace::Device>;

using HostCsrMatrix   = CsrMatrix<MemorySpace::Host>;
using DeviceCsrMatrix = CsrMatrix<MemorySpace::Device>;

template <class T>
using HostArrayView = VectorView<MemorySpace::Host, T>;

using HostVectorView      = HostArrayView<Real>;
using HostConstVectorView = HostArrayView<const Real>;

namespace constants
{
constexpr Real ZERO      = 0.0;
constexpr Real ONE       = 1.0;
constexpr Real TWO       = 2.0;
constexpr Real HALF      = 0.5;
constexpr Real MINUS_ONE = -1.0;
constexpr Real PI        = 3.141592653589793238462643383279502884;

constexpr Real MACHINE_EPSILON = std::numeric_limits<Real>::epsilon();
} // namespace constants

namespace colors
{
// must be const pointer and const dest for
// const string declarations to pass -Wwrite-strings
static const char* const RED    = "\033[1;31m";
static const char* const GREEN  = "\033[1;32m";
static const char* const YELLOW = "\033[33;1m";
static const char* const BLUE   = "\033[34;1m";
static const char* const ORANGE = "\u001b[38;5;208m";
static const char* const CLEAR  = "\033[0m";
} // namespace colors

} // namespace femx

#if defined(__CUDACC__)
#define FEMX_HOST_DEVICE __host__ __device__
#define FEMX_DEVICE      __device__
#else
#define FEMX_HOST_DEVICE
#define FEMX_DEVICE
#endif
