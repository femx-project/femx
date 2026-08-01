#pragma once

#include <algorithm>
#include <cstddef>

#include <femx/common/Checks.hpp>
#include <femx/common/Vector.hpp>

namespace femx::linalg
{

/**
 * @brief Define vector operations for one memory space.
 */
template <MemorySpace Space>
class VectorHandler;

/**
 * @brief Define backend-independent Host vector operations.
 */
template <>
class VectorHandler<MemorySpace::Host>
{
public:
  virtual ~VectorHandler() = default;

  /**
   * @brief Copy between same-sized Host views.
   *
   * @param[in]  src - Source view.
   * @param[out] dst - Destination view.
   * @throws - If sizes differ or views partially overlap.
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
   */
  template <class T>
  void copy(HostVectorView<T> src, HostVectorView<T> dst) const
  {
    copy(HostVectorView<const T>(src.data(), src.size()), dst);
  }

  /**
   * @brief Replace a Host vector by copying a view.
   */
  template <class T>
  void copy(HostVectorView<const T> src, HostVector<T>& dst) const
  {
    dst = src;
  }

  /**
   * @brief Replace a Host vector by copying a mutable view.
   */
  template <class T>
  void copy(HostVectorView<T> src, HostVector<T>& dst) const
  {
    copy(HostVectorView<const T>(src.data(), src.size()), dst);
  }

  /**
   * @brief Replace a Host vector with copies of one value.
   */
  template <class T>
  void assign(HostVector<T>&                            out,
              Index                                     size,
              const typename HostVector<T>::value_type& val) const
  {
    out.assign(size, val);
  }

  virtual void zero(HostVectorView<Real> vals) const = 0;

  virtual void axpby(Real                       a,
                     HostVectorView<const Real> x,
                     Real                       b,
                     HostVectorView<Real>       y) const = 0;

  virtual Real dot(HostVectorView<const Real> x,
                   HostVectorView<const Real> y) const = 0;

  Real squaredNorm(HostVectorView<const Real> x) const
  {
    return dot(x, x);
  }

  virtual void gather(HostVectorView<const Real>  src,
                      HostVectorView<const Index> indices,
                      HostVectorView<Real>        dst) const = 0;

  virtual void scatter(HostVectorView<const Real>  src,
                       HostVectorView<const Index> indices,
                       HostVectorView<Real>        dst) const = 0;
};

/**
 * @brief Define backend-independent Device vector operations.
 */
template <>
class VectorHandler<MemorySpace::Device>
{
public:
  virtual ~VectorHandler() = default;

  /**
   * @brief Copy a Host vector to Device storage.
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
   * @brief Copy a Device vector to Device storage.
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
   * @brief Copy a Device vector to Host storage.
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

  template <class T>
  void copy(HostVector<T>&&, DeviceVector<T>&) const = delete;

  template <class T>
  void copy(DeviceVector<T>&&, DeviceVector<T>&) const = delete;

  virtual void assign(DeviceVector<Real>& out,
                      Index               size,
                      Real                val) const = 0;

  virtual void assign(DeviceVector<Index>& out,
                      Index                size,
                      Index                val) const = 0;

  virtual void copy(DeviceVectorView<const Real> src,
                    DeviceVectorView<Real>       dst) const = 0;

  virtual void copy(DeviceVectorView<const Real> src,
                    DeviceVector<Real>&          dst) const = 0;

  virtual void copy(HostVectorView<const Real> src,
                    DeviceVectorView<Real>     dst) const = 0;

  virtual void copy(HostVectorView<const Real> src,
                    DeviceVector<Real>&        dst) const = 0;

  virtual void copy(DeviceVectorView<const Real> src,
                    HostVectorView<Real>         dst) const = 0;

  virtual void copy(DeviceVectorView<const Real> src,
                    HostVector<Real>&            dst) const = 0;

  virtual void zero(DeviceVectorView<Real> vals) const = 0;

  virtual void axpby(Real                         a,
                     DeviceVectorView<const Real> x,
                     Real                         b,
                     DeviceVectorView<Real>       y) const = 0;

  virtual void gather(DeviceVectorView<const Real>  src,
                      DeviceVectorView<const Index> indices,
                      DeviceVectorView<Real>        dst) const = 0;

  virtual void scatter(DeviceVectorView<const Real>  src,
                       DeviceVectorView<const Index> indices,
                       DeviceVectorView<Real>        dst) const = 0;

  virtual void dot(DeviceVectorView<const Real> x,
                   DeviceVectorView<const Real> y,
                   DeviceVectorView<Real>       out) const = 0;

  void squaredNorm(DeviceVectorView<const Real> x,
                   DeviceVectorView<Real>       out) const
  {
    dot(x, x, out);
  }

protected:
  virtual void copyBytes(const void* src,
                         MemorySpace src_space,
                         void*       dst,
                         MemorySpace dst_space,
                         std::size_t bytes) const = 0;

  template <class T>
  static void resize(DeviceVector<T>& dst, Index size)
  {
    if (dst.size() != size)
    {
      dst.resize(size);
    }
  }

  template <class T>
  static void resize(HostVector<T>& dst, Index size)
  {
    if (dst.size() != size)
    {
      dst.resize(size);
    }
  }

private:
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
