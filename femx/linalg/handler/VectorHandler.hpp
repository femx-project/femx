#pragma once

#include <algorithm>
#include <cstddef>

#include <femx/common/Checks.hpp>
#include <femx/linalg/Backend.hpp>

namespace femx::linalg
{

/** @brief Provide vector operations for an execution backend. */
template <class Backend>
class VectorHandler;

/** @brief Provide serial CPU vector operations. */
template <>
class VectorHandler<HostCsrBackend> final
{
public:
  /**
   * @brief Bind vector operations to a CPU context.
   *
   * @param[in] ctx - CPU execution context.
   */
  explicit VectorHandler(CpuContext& ctx) noexcept
    : ctx_(ctx)
  {
  }

  /**
   * @brief Copy between same-sized Host views.
   *
   * @param[in] src - Source view.
   * @param[out] dst - Destination view.
   * @throws std::runtime_error - If sizes differ or views partially overlap.
   */
  template <class T>
  void copy(VectorView<MemorySpace::Host, const T> src,
            VectorView<MemorySpace::Host, T>       dst) const
  {
    require(src.size() == dst.size(),
            "Host view copy requires equal sizes");
    if (src.empty() || src.data() == dst.data())
    {
      return;
    }
    require(!femx::detail::overlaps(src, dst),
            "Host view copy does not support partial overlap");
    std::copy(src.begin(), src.end(), dst.begin());
  }

  /**
   * @brief Copy between same-sized mutable Host views.
   *
   * @param[in] src - Source view.
   * @param[out] dst - Destination view.
   * @throws std::runtime_error - If sizes differ or views partially overlap.
   */
  template <class T>
  void copy(VectorView<MemorySpace::Host, T> src,
            VectorView<MemorySpace::Host, T> dst) const
  {
    copy(VectorView<MemorySpace::Host, const T>(src.data(), src.size()),
         dst);
  }

  /**
   * @brief Replace a Host vector by copying a view.
   *
   * @param[in] src - Source view.
   * @param[out] dst - Destination vector.
   * @throws std::runtime_error - If the view size is negative.
   */
  template <class T>
  void copy(VectorView<MemorySpace::Host, const T> src,
            Vector<MemorySpace::Host, T>&          dst) const
  {
    dst = src;
  }

  /**
   * @brief Replace a Host vector by copying a mutable view.
   *
   * @param[in] src - Source view.
   * @param[out] dst - Destination vector.
   * @throws std::runtime_error - If the view size is negative.
   */
  template <class T>
  void copy(VectorView<MemorySpace::Host, T> src,
            Vector<MemorySpace::Host, T>&    dst) const
  {
    copy(VectorView<MemorySpace::Host, const T>(src.data(), src.size()),
         dst);
  }

  /**
   * @brief Resize a Host vector if needed and set every value to zero.
   *
   * @param[in,out] out - Vector to resize or clear.
   * @param[in] size - Required vector size.
   * @throws std::runtime_error - If `size` is negative.
   */
  template <class T>
  void resizeOrZero(Vector<MemorySpace::Host, T>& out, Index size) const
  {
    if (out.size() != size)
    {
      out.resize(size);
    }
    else
    {
      std::fill(out.begin(), out.end(), T{});
    }
  }

  /**
   * @brief Set every value to zero.
   *
   * @param[out] vals - Values to clear.
   */
  void zero(HostVectorView vals) const;

  /**
   * @brief Compute `y = a * x + b * y`.
   *
   * @param[in] a - Input-vector scale.
   * @param[in] x - Input vector.
   * @param[in] b - Existing-output scale.
   * @param[in,out] y - Output vector.
   * @throws std::runtime_error - If sizes or storage overlap are invalid.
   */
  void axpby(Real                a,
             HostConstVectorView x,
             Real                b,
             HostVectorView      y) const;

  /**
   * @brief Compute the dot product of two vectors.
   *
   * @param[in] x - First input vector.
   * @param[in] y - Second input vector.
   * @return Dot product of `x` and `y`.
   * @throws std::runtime_error - If vector sizes differ.
   */
  Real dot(HostConstVectorView x, HostConstVectorView y) const;

  /**
   * @brief Compute the squared Euclidean norm of a vector.
   *
   * @param[in] x - Input vector.
   * @return Squared Euclidean norm of `x`.
   */
  Real squaredNorm(HostConstVectorView x) const;

  /**
   * @brief Gather indexed source values into a contiguous destination.
   *
   * @param[in] src - Source values.
   * @param[in] indices - Source indices in destination order.
   * @param[out] dst - Contiguous destination values.
   * @throws std::runtime_error - If sizes, indices, or aliasing are invalid.
   */
  void gather(HostConstVectorView src,
              HostConstIndexView  indices,
              HostVectorView      dst) const;

  /**
   * @brief Scatter contiguous source values to indexed destinations.
   *
   * @param[in] src - Contiguous source values.
   * @param[in] indices - Destination indices in source order.
   * @param[out] dst - Indexed destination values.
   * @throws std::runtime_error - If sizes, indices, or aliasing are invalid.
   */
  void scatter(HostConstVectorView src,
               HostConstIndexView  indices,
               HostVectorView      dst) const;

private:
  CpuContext& ctx_; ///< Bound CPU context.
};

/**
 * @brief Provide CUDA vector operations and explicit Host/Device transfers.
 *
 * Operations are enqueued on the stream owned by the bound context.
 * Synchronize the context before reading Host destinations.
 */
template <>
class VectorHandler<CudaCsrBackend> final
{
public:
  /**
   * @brief Bind vector operations to a CUDA context.
   *
   * @param[in] ctx - CUDA execution context.
   */
  explicit VectorHandler(CudaContext& ctx) noexcept
    : ctx_(ctx)
  {
  }

  /**
   * @brief Copy a Host vector to Device storage.
   *
   * @param[in] src - Source Host vector.
   * @param[out] dst - Destination Device vector.
   * @throws std::runtime_error - If allocation or a CUDA operation fails.
   */
  template <class T>
  void copy(const Vector<MemorySpace::Host, T>& src,
            Vector<MemorySpace::Device, T>&     dst) const
  {
    resize(dst, src.size());
    copyStorage(src.data(), MemorySpace::Host, dst.data(), src.size());
  }

  /**
   * @brief Copy a Device vector to Device storage.
   *
   * @param[in] src - Source Device vector.
   * @param[out] dst - Destination Device vector.
   * @throws std::runtime_error - If allocation or a CUDA operation fails.
   */
  template <class T>
  void copy(const Vector<MemorySpace::Device, T>& src,
            Vector<MemorySpace::Device, T>&       dst) const
  {
    if (&src == &dst)
    {
      return;
    }
    resize(dst, src.size());
    copyStorage(src.data(), MemorySpace::Device, dst.data(), src.size());
  }

  /**
   * @brief Copy a Device vector to Host storage.
   *
   * @param[in] src - Source Device vector.
   * @param[out] dst - Destination Host vector.
   * @throws std::runtime_error - If allocation or a CUDA operation fails.
   */
  template <class T>
  void copy(const Vector<MemorySpace::Device, T>& src,
            Vector<MemorySpace::Host, T>&         dst) const
  {
    resize(dst, src.size());
    if (!src.empty())
    {
      cuda::copy(dst.data(),
                 MemorySpace::Host,
                 src.data(),
                 MemorySpace::Device,
                 static_cast<std::size_t>(src.size()) * sizeof(T),
                 ctx_.stream());
    }
  }

  /**
   * @brief Reject copying from a temporary Host vector.
   *
   * @param[in] src - Temporary source vector.
   * @param[out] dst - Destination vector.
   */
  template <class T>
  void copy(Vector<MemorySpace::Host, T>&&  src,
            Vector<MemorySpace::Device, T>& dst) const = delete;

  /**
   * @brief Reject copying from a temporary Device vector.
   *
   * @param[in] src - Temporary source vector.
   * @param[out] dst - Destination vector.
   */
  template <class T>
  void copy(Vector<MemorySpace::Device, T>&& src,
            Vector<MemorySpace::Device, T>&  dst) const = delete;

  /**
   * @brief Resize a Device vector if needed and set every value to zero.
   *
   * @param[in,out] out - Device vector to resize or clear.
   * @param[in] size - Required vector size.
   * @throws std::runtime_error - If `size` is negative or a CUDA operation
   * fails.
   */
  template <class T>
  void resizeOrZero(Vector<MemorySpace::Device, T>& out, Index size) const
  {
    if (out.size() != size)
    {
      out.resize(size);
    }
    else if (!out.empty())
    {
      cuda::zero(out.data(),
                 static_cast<std::size_t>(out.size()) * sizeof(T),
                 ctx_.stream());
    }
  }

  /**
   * @brief Copy between same-sized Device views.
   *
   * @param[in] src - Source Device view.
   * @param[out] dst - Destination Device view.
   * @throws std::runtime_error - If views are invalid, sizes differ, overlap,
   * or a CUDA operation fails.
   */
  void copy(DeviceConstVectorView src, DeviceVectorView dst) const;

  /**
   * @brief Replace a Device vector by copying a Device view.
   *
   * @param[in] src - Source Device view.
   * @param[out] dst - Destination Device vector.
   * @throws std::runtime_error - If the view is invalid or a CUDA operation
   * fails.
   */
  void copy(DeviceConstVectorView src, DeviceVector& dst) const;

  /**
   * @brief Copy between same-sized Host and Device views.
   *
   * @param[in] src - Source Host view.
   * @param[out] dst - Destination Device view.
   * @throws std::runtime_error - If views are invalid, sizes differ, or a CUDA
   * operation fails.
   */
  void copy(HostConstVectorView src, DeviceVectorView dst) const;

  /**
   * @brief Replace a Device vector by copying a Host view.
   *
   * @param[in] src - Source Host view.
   * @param[out] dst - Destination Device vector.
   * @throws std::runtime_error - If the view is invalid or a CUDA operation
   * fails.
   */
  void copy(HostConstVectorView src, DeviceVector& dst) const;

  /**
   * @brief Copy between same-sized Device and Host views.
   *
   * @param[in] src - Source Device view.
   * @param[out] dst - Destination Host view.
   * @throws std::runtime_error - If views are invalid, sizes differ, or a CUDA
   * operation fails.
   */
  void copy(DeviceConstVectorView src, HostVectorView dst) const;

  /**
   * @brief Replace a Host vector by copying a Device view.
   *
   * @param[in] src - Source Device view.
   * @param[out] dst - Destination Host vector.
   * @throws std::runtime_error - If the view is invalid or a CUDA operation
   * fails.
   */
  void copy(DeviceConstVectorView src, HostVector& dst) const;

  /**
   * @brief Set every Device value to zero.
   *
   * @param[out] vals - Device values to clear.
   * @throws std::runtime_error - If the view is invalid or a CUDA operation
   * fails.
   */
  void zero(DeviceVectorView vals) const;

  /**
   * @brief Compute `y = a * x + b * y` on Device.
   *
   * @param[in] a - Input-vector scale.
   * @param[in] x - Device input vector.
   * @param[in] b - Existing-output scale.
   * @param[in,out] y - Device output vector.
   * @throws std::runtime_error - If inputs are invalid or a CUDA operation
   * fails.
   */
  void axpby(Real                  a,
             DeviceConstVectorView x,
             Real                  b,
             DeviceVectorView      y) const;

  /**
   * @brief Gather indexed Device values into a contiguous destination.
   *
   * @param[in] src - Device source values.
   * @param[in] indices - Device source indices in destination order.
   * @param[out] dst - Contiguous Device destination values.
   * @throws std::runtime_error - If inputs are invalid or a CUDA operation
   * fails.
   */
  void gather(DeviceConstVectorView src,
              DeviceConstIndexView  indices,
              DeviceVectorView      dst) const;

  /**
   * @brief Scatter contiguous Device values to indexed destinations.
   *
   * @param[in] src - Contiguous Device source values.
   * @param[in] indices - Device destination indices in source order.
   * @param[out] dst - Indexed Device destination values.
   * @throws std::runtime_error - If inputs are invalid or a CUDA operation
   * fails.
   */
  void scatter(DeviceConstVectorView src,
               DeviceConstIndexView  indices,
               DeviceVectorView      dst) const;

  /**
   * @brief Compute a Device dot product into one Device value.
   *
   * @param[in] x - First Device input vector.
   * @param[in] y - Second Device input vector.
   * @param[out] out - One-value Device result view.
   * @throws std::runtime_error - If inputs are invalid or a CUDA operation
   * fails.
   */
  void dot(DeviceConstVectorView x,
           DeviceConstVectorView y,
           DeviceVectorView      out) const;

  /**
   * @brief Compute a squared Euclidean norm into one Device value.
   *
   * @param[in] x - Device input vector.
   * @param[out] out - One-value Device result view.
   * @throws std::runtime_error - If inputs are invalid or a CUDA operation
   * fails.
   */
  void squaredNorm(DeviceConstVectorView x, DeviceVectorView out) const
  {
    dot(x, x, out);
  }

private:
  template <class T>
  static void resize(Vector<MemorySpace::Device, T>& dst, Index size)
  {
    if (dst.size() != size)
    {
      dst.resize(size);
    }
  }

  template <class T>
  static void resize(Vector<MemorySpace::Host, T>& dst, Index size)
  {
    if (dst.size() != size)
    {
      dst.resize(size);
    }
  }

  template <class T>
  void copyStorage(const T*    src,
                   MemorySpace src_space,
                   T*          dst,
                   Index       size) const
  {
    if (size > 0)
    {
      cuda::copy(dst,
                 MemorySpace::Device,
                 src,
                 src_space,
                 static_cast<std::size_t>(size) * sizeof(T),
                 ctx_.stream());
    }
  }

  CudaContext& ctx_; ///< Bound CUDA context.
};

using HostVectorHandler = VectorHandler<HostCsrBackend>;
using CudaVectorHandler = VectorHandler<CudaCsrBackend>;

} // namespace femx::linalg
