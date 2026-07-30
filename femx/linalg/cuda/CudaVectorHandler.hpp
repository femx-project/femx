#pragma once

#include <cstddef>

#include <femx/common/Cuda.hpp>
#include <femx/common/Vector.hpp>

namespace femx::linalg
{

class CudaContext;

/**
 * @brief Provide CUDA vector operations and explicit Host/Device transfers.
 *
 * Operations are enqueued on the stream owned by the bound context.
 * Synchronize the context before reading Host destinations.
 */
class CudaVectorHandler final
{
public:
  /**
   * @brief Bind vector operations to a CUDA context.
   *
   * @param[in] ctx - CUDA execution context.
   */
  explicit CudaVectorHandler(CudaContext& ctx) noexcept;

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
                 stream());
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
   * @brief Replace a Device vector with copies of one value.
   *
   * @param[out] out - Device vector to replace.
   * @param[in] size - Required vector size.
   * @param[in] val - Value assigned to every entry.
   * @throws std::runtime_error - If `size` is negative or a CUDA operation
   * fails.
   */
  void assign(DeviceVector<Real>& out, Index size, Real val) const;

  /**
   * @brief Replace a Device index vector with copies of one value.
   *
   * @param[out] out - Device vector to replace.
   * @param[in] size - Required vector size.
   * @param[in] val - Value assigned to every entry.
   * @throws std::runtime_error - If `size` is negative or a CUDA operation
   * fails.
   */
  void assign(DeviceVector<Index>& out, Index size, Index val) const;

  /**
   * @brief Copy between same-sized Device views.
   *
   * @param[in] src - Source Device view.
   * @param[out] dst - Destination Device view.
   * @throws std::runtime_error - If views are invalid, sizes differ, overlap,
   * or a CUDA operation fails.
   */
  void copy(DeviceVectorView<const Real> src, DeviceVectorView<Real> dst) const;

  /**
   * @brief Replace a Device vector by copying a Device view.
   *
   * @param[in] src - Source Device view.
   * @param[out] dst - Destination Device vector.
   * @throws std::runtime_error - If the view is invalid or a CUDA operation
   * fails.
   */
  void copy(DeviceVectorView<const Real> src, DeviceVector<Real>& dst) const;

  /**
   * @brief Copy between same-sized Host and Device views.
   *
   * @param[in] src - Source Host view.
   * @param[out] dst - Destination Device view.
   * @throws std::runtime_error - If views are invalid, sizes differ, or a CUDA
   * operation fails.
   */
  void copy(HostVectorView<const Real> src, DeviceVectorView<Real> dst) const;

  /**
   * @brief Replace a Device vector by copying a Host view.
   *
   * @param[in] src - Source Host view.
   * @param[out] dst - Destination Device vector.
   * @throws std::runtime_error - If the view is invalid or a CUDA operation
   * fails.
   */
  void copy(HostVectorView<const Real> src, DeviceVector<Real>& dst) const;

  /**
   * @brief Copy between same-sized Device and Host views.
   *
   * @param[in] src - Source Device view.
   * @param[out] dst - Destination Host view.
   * @throws std::runtime_error - If views are invalid, sizes differ, or a CUDA
   * operation fails.
   */
  void copy(DeviceVectorView<const Real> src, HostVectorView<Real> dst) const;

  /**
   * @brief Replace a Host vector by copying a Device view.
   *
   * @param[in] src - Source Device view.
   * @param[out] dst - Destination Host vector.
   * @throws std::runtime_error - If the view is invalid or a CUDA operation
   * fails.
   */
  void copy(DeviceVectorView<const Real> src, HostVector<Real>& dst) const;

  /**
   * @brief Set every Device value to zero.
   *
   * @param[out] vals - Device values to clear.
   * @throws std::runtime_error - If the view is invalid or a CUDA operation
   * fails.
   */
  void zero(DeviceVectorView<Real> vals) const;

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
  void axpby(Real                         a,
             DeviceVectorView<const Real> x,
             Real                         b,
             DeviceVectorView<Real>       y) const;

  /**
   * @brief Gather indexed Device values into a contiguous destination.
   *
   * @param[in] src - Device source values.
   * @param[in] indices - Device source indices in destination order.
   * @param[out] dst - Contiguous Device destination values.
   * @throws std::runtime_error - If inputs are invalid or a CUDA operation
   * fails.
   */
  void gather(DeviceVectorView<const Real>  src,
              DeviceVectorView<const Index> indices,
              DeviceVectorView<Real>        dst) const;

  /**
   * @brief Scatter contiguous Device values to indexed destinations.
   *
   * @param[in] src - Contiguous Device source values.
   * @param[in] indices - Device destination indices in source order.
   * @param[out] dst - Indexed Device destination values.
   * @throws std::runtime_error - If inputs are invalid or a CUDA operation
   * fails.
   */
  void scatter(DeviceVectorView<const Real>  src,
               DeviceVectorView<const Index> indices,
               DeviceVectorView<Real>        dst) const;

  /**
   * @brief Compute a Device dot product into one Device value.
   *
   * @param[in] x - First Device input vector.
   * @param[in] y - Second Device input vector.
   * @param[out] out - One-value Device result view.
   * @throws std::runtime_error - If inputs are invalid or a CUDA operation
   * fails.
   */
  void dot(DeviceVectorView<const Real> x,
           DeviceVectorView<const Real> y,
           DeviceVectorView<Real>       out) const;

  /**
   * @brief Compute a squared Euclidean norm into one Device value.
   *
   * @param[in] x - Device input vector.
   * @param[out] out - One-value Device result view.
   * @throws std::runtime_error - If inputs are invalid or a CUDA operation
   * fails.
   */
  void squaredNorm(DeviceVectorView<const Real> x,
                   DeviceVectorView<Real>       out) const
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
                 stream());
    }
  }

  void* stream() const noexcept;

  CudaContext& ctx_; ///< Bound CUDA context.
};

} // namespace femx::linalg
