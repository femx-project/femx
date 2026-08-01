#pragma once

#include <algorithm>
#include <cstddef>

#include <femx/common/Checks.hpp>
#include <femx/common/Vector.hpp>

namespace femx::linalg
{

/**
 * @brief Define vector operations for one memory space.
 *
 * @tparam Space - Storage location on which the operations act.
 */
template <MemorySpace Space>
class VectorHandler;

/**
 * @brief Define backend-independent operations on Host vectors and views.
 */
template <>
class VectorHandler<MemorySpace::Host>
{
public:
  virtual ~VectorHandler() = default;

  /**
   * @brief Copy between same-sized Host views.
   *
   * @tparam T - Copied value type.
   * @param[in]  src - Source view.
   * @param[out] dst - Destination view.
   * @throws std::runtime_error If validation fails.
   */
  template <class T>
  void copy(HostVectorView<const T> src, HostVectorView<T> dst) const
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
   * @tparam T - Copied value type.
   * @param[in]  src - Mutable source view read by the operation.
   * @param[out] dst - Destination view.
   * @throws std::runtime_error If validation fails.
   */
  template <class T>
  void copy(HostVectorView<T> src, HostVectorView<T> dst) const
  {
    copy(HostVectorView<const T>(src.data(), src.size()), dst);
  }

  /**
   * @brief Resize a Host vector and copy a view into it.
   *
   * @tparam T - Copied value type.
   * @param[in]  src - Source view.
   * @param[out] dst - Owning destination replaced by the copied values.
   */
  template <class T>
  void copy(HostVectorView<const T> src, HostVector<T>& dst) const
  {
    dst = src;
  }

  /**
   * @brief Resize a Host vector and copy a mutable view into it.
   *
   * @tparam T - Copied value type.
   * @param[in]  src - Mutable source view read by the operation.
   * @param[out] dst - Owning destination replaced by the copied values.
   */
  template <class T>
  void copy(HostVectorView<T> src, HostVector<T>& dst) const
  {
    copy(HostVectorView<const T>(src.data(), src.size()), dst);
  }

  /**
   * @brief Replace a Host vector with copies of one value.
   *
   * @tparam T - Assigned value type.
   * @param[out] out  - Owning vector resized and filled by the operation.
   * @param[in]  size - Number of output values.
   * @param[in]  val  - Value assigned to every output entry.
   * @throws std::runtime_error If validation fails.
   */
  template <class T>
  void assign(HostVector<T>&                            out,
              Index                                     size,
              const typename HostVector<T>::value_type& val) const
  {
    out.assign(size, val);
  }

  /**
   * @brief Set every value in a Host view to zero.
   *
   * @param[out] vals - Values cleared in place.
   */
  virtual void zero(HostVectorView<Real> vals) const = 0;

  /**
   * @brief Compute `y = a * x + b * y` on Host.
   *
   * @param[in]     a - Scale applied to `x`.
   * @param[in]     x - Input vector.
   * @param[in]     b - Scale applied to the previous contents of `y`.
   * @param[in,out] y - Input/output vector.
   * @throws std::runtime_error If validation fails.
   */
  virtual void axpby(Real                       a,
                     HostVectorView<const Real> x,
                     Real                       b,
                     HostVectorView<Real>       y) const = 0;

  /**
   * @brief Compute the Host dot product `sum(x[i] * y[i])`.
   *
   * @param[in] x - First input vector.
   * @param[in] y - Second input vector.
   * @return Dot-product value.
   * @throws std::runtime_error If validation fails.
   */
  virtual Real dot(HostVectorView<const Real> x,
                   HostVectorView<const Real> y) const = 0;

  /**
   * @brief Compute the squared Euclidean norm `dot(x, x)` on Host.
   *
   * @param[in] x - Input vector.
   * @return Squared Euclidean norm.
   */
  Real squaredNorm(HostVectorView<const Real> x) const
  {
    return dot(x, x);
  }

  /**
   * @brief Gather indexed Host values into a contiguous destination.
   *
   * The operation computes `dst[i] = src[idx[i]]`.
   *
   * @param[in]  src - Source values.
   * @param[in]  idx - Source indices in destination order.
   * @param[out] dst - Contiguous destination with `idx.size()` entries.
   * @throws std::runtime_error If validation fails.
   */
  virtual void gather(HostVectorView<const Real>  src,
                      HostVectorView<const Index> idx,
                      HostVectorView<Real>        dst) const = 0;

  /**
   * @brief Scatter contiguous Host values to indexed destinations.
   *
   * The operation computes `dst[idx[i]] = src[i]`; destination entries not
   * referenced by `idx` remain unchanged.
   *
   * @param[in]     src - Contiguous source values.
   * @param[in]     idx - Destination indices in source order.
   * @param[in,out] dst - Indexed destination values.
   * @throws std::runtime_error If validation fails.
   */
  virtual void scatter(HostVectorView<const Real>  src,
                       HostVectorView<const Index> idx,
                       HostVectorView<Real>        dst) const = 0;
};

/**
 * @brief Define backend-independent operations and transfers for Device data.
 */
template <>
class VectorHandler<MemorySpace::Device>
{
public:
  virtual ~VectorHandler() = default;

  /**
   * @brief Resize Device storage and copy an owning Host vector into it.
   *
   * @tparam T - Trivially copyable value type.
   * @param[in]  src - Owning Host source.
   * @param[out] dst - Owning Device destination resized to `src.size()`.
   * @throws std::runtime_error If validation fails.
   */
  template <class T>
  void copy(const HostVector<T>& src, DeviceVector<T>& dst) const
  {
    resize(dst, src.size());
    copyStorage(src.data(),
                MemorySpace::Host,
                dst.data(),
                MemorySpace::Device,
                src.size());
  }

  /**
   * @brief Resize Device storage and copy an owning Device vector into it.
   *
   * @tparam T - Trivially copyable value type.
   * @param[in]  src - Owning Device source.
   * @param[out] dst - Owning Device destination resized to `src.size()`.
   * @throws std::runtime_error If validation fails.
   */
  template <class T>
  void copy(const DeviceVector<T>& src, DeviceVector<T>& dst) const
  {
    if (&src == &dst)
    {
      return;
    }
    resize(dst, src.size());
    copyStorage(src.data(),
                MemorySpace::Device,
                dst.data(),
                MemorySpace::Device,
                src.size());
  }

  /**
   * @brief Resize Host storage and copy an owning Device vector into it.
   *
   * @tparam T - Trivially copyable value type.
   * @param[in]  src - Owning Device source.
   * @param[out] dst - Owning Host destination resized to `src.size()`.
   * @throws std::runtime_error If validation fails.
   */
  template <class T>
  void copy(const DeviceVector<T>& src, HostVector<T>& dst) const
  {
    resize(dst, src.size());
    copyStorage(src.data(),
                MemorySpace::Device,
                dst.data(),
                MemorySpace::Host,
                src.size());
  }

  /**
   * @brief Reject an asynchronous copy from a temporary Host vector.
   */
  template <class T>
  void copy(HostVector<T>&&, DeviceVector<T>&) const = delete;

  /**
   * @brief Reject an asynchronous copy from a temporary Device vector.
   */
  template <class T>
  void copy(DeviceVector<T>&&, DeviceVector<T>&) const = delete;

  /**
   * @brief Resize a Device real vector and fill it with one value.
   *
   * @param[out] out  - Owning Device vector resized and filled in place.
   * @param[in]  size - Number of output values.
   * @param[in]  val  - Value assigned to every output entry.
   * @throws std::runtime_error If validation fails.
   */
  virtual void assign(DeviceVector<Real>& out,
                      Index               size,
                      Real                val) const = 0;

  /**
   * @brief Resize a Device index vector and fill it with one value.
   *
   * @param[out] out  - Owning Device vector resized and filled in place.
   * @param[in]  size - Number of output indices.
   * @param[in]  val  - Value assigned to every output entry.
   * @throws std::runtime_error If validation fails.
   */
  virtual void assign(DeviceVector<Index>& out,
                      Index                size,
                      Index                val) const = 0;

  /**
   * @brief Copy between same-sized Device views.
   *
   * @param[in]  src - Device source view.
   * @param[out] dst - Device destination view.
   * @throws std::runtime_error If validation fails.
   */
  virtual void copy(DeviceVectorView<const Real> src,
                    DeviceVectorView<Real>       dst) const = 0;

  /**
   * @brief Resize an owning Device vector and copy a Device view into it.
   *
   * @param[in]  src - Device source view.
   * @param[out] dst - Owning Device destination resized to `src.size()`.
   * @throws std::runtime_error If validation fails.
   */
  virtual void copy(DeviceVectorView<const Real> src,
                    DeviceVector<Real>&          dst) const = 0;

  /**
   * @brief Copy a Host view into a same-sized Device view.
   *
   * @param[in]  src - Host source view.
   * @param[out] dst - Device destination view.
   * @throws std::runtime_error If validation fails.
   */
  virtual void copy(HostVectorView<const Real> src,
                    DeviceVectorView<Real>     dst) const = 0;

  /**
   * @brief Resize an owning Device vector and copy a Host view into it.
   *
   * @param[in]  src - Host source view.
   * @param[out] dst - Owning Device destination resized to `src.size()`.
   * @throws std::runtime_error If validation fails.
   */
  virtual void copy(HostVectorView<const Real> src,
                    DeviceVector<Real>&        dst) const = 0;

  /**
   * @brief Copy a Device view into a same-sized Host view.
   *
   * @param[in]  src - Device source view.
   * @param[out] dst - Host destination view.
   * @throws std::runtime_error If validation fails.
   */
  virtual void copy(DeviceVectorView<const Real> src,
                    HostVectorView<Real>         dst) const = 0;

  /**
   * @brief Resize an owning Host vector and copy a Device view into it.
   *
   * @param[in]  src - Device source view.
   * @param[out] dst - Owning Host destination resized to `src.size()`.
   * @throws std::runtime_error If validation fails.
   */
  virtual void copy(DeviceVectorView<const Real> src,
                    HostVector<Real>&            dst) const = 0;

  /**
   * @brief Set every value in a Device view to zero.
   *
   * @param[out] vals - Device values cleared in place.
   * @throws std::runtime_error If validation fails.
   */
  virtual void zero(DeviceVectorView<Real> vals) const = 0;

  /**
   * @brief Compute `y = a * x + b * y` on Device.
   *
   * @param[in]     a - Scale applied to `x`.
   * @param[in]     x - Device input vector.
   * @param[in]     b - Scale applied to the previous contents of `y`.
   * @param[in,out] y - Device input/output vector.
   * @throws std::runtime_error If validation fails.
   */
  virtual void axpby(Real                         a,
                     DeviceVectorView<const Real> x,
                     Real                         b,
                     DeviceVectorView<Real>       y) const = 0;

  /**
   * @brief Gather indexed Device values into a contiguous destination.
   *
   * @param[in]  src - Device source values.
   * @param[in]  idx - Device source indices in destination order.
   * @param[out] dst - Contiguous Device destination.
   * @throws std::runtime_error If validation fails.
   */
  virtual void gather(DeviceVectorView<const Real>  src,
                      DeviceVectorView<const Index> idx,
                      DeviceVectorView<Real>        dst) const = 0;

  /**
   * @brief Scatter contiguous Device values to indexed destinations.
   *
   * @param[in]     src - Contiguous Device source values.
   * @param[in]     idx - Device destination indices in source order.
   * @param[in,out] dst - Indexed Device destination values.
   * @throws std::runtime_error If validation fails.
   */
  virtual void scatter(DeviceVectorView<const Real>  src,
                       DeviceVectorView<const Index> idx,
                       DeviceVectorView<Real>        dst) const = 0;

  /**
   * @brief Compute a Device dot product into one Device scalar.
   *
   * @param[in]  x   - First Device input vector.
   * @param[in]  y   - Second Device input vector.
   * @param[out] out - Single-entry Device view receiving the result.
   * @throws std::runtime_error If validation fails.
   */
  virtual void dot(DeviceVectorView<const Real> x,
                   DeviceVectorView<const Real> y,
                   DeviceVectorView<Real>       out) const = 0;

  /**
   * @brief Compute the squared Euclidean norm into one Device scalar.
   *
   * @param[in]  x   - Device input vector.
   * @param[out] out - Single-entry Device view receiving `dot(x, x)`.
   * @throws std::runtime_error If validation fails.
   */
  void squaredNorm(DeviceVectorView<const Real> x,
                   DeviceVectorView<Real>       out) const
  {
    dot(x, x, out);
  }

protected:
  /**
   * @brief Copy raw bytes between Host or Device memory spaces.
   *
   * @param[in]  src       - Source address.
   * @param[in]  src_space - Memory space containing `src`.
   * @param[out] dst       - Destination address.
   * @param[in]  dst_space - Memory space containing `dst`.
   * @param[in]  bytes     - Number of bytes to copy.
   * @throws std::runtime_error If validation fails.
   */
  virtual void copyBytes(const void* src,
                         MemorySpace src_space,
                         void*       dst,
                         MemorySpace dst_space,
                         std::size_t bytes) const = 0;

  /**
   * @brief Resize an owning Device vector only when its size differs.
   */
  template <class T>
  static void resize(DeviceVector<T>& dst, Index size)
  {
    if (dst.size() != size)
    {
      dst.resize(size);
    }
  }

  /**
   * @brief Resize an owning Host vector only when its size differs.
   */
  template <class T>
  static void resize(HostVector<T>& dst, Index size)
  {
    if (dst.size() != size)
    {
      dst.resize(size);
    }
  }

private:
  /**
   * @brief Copy a typed contiguous allocation through `copyBytes()`.
   */
  template <class T>
  void copyStorage(const T*    src,
                   MemorySpace src_space,
                   T*          dst,
                   MemorySpace dst_space,
                   Index       size) const
  {
    if (size > 0)
    {
      copyBytes(src,
                src_space,
                dst,
                dst_space,
                static_cast<std::size_t>(size) * sizeof(T));
    }
  }
};

} // namespace femx::linalg
