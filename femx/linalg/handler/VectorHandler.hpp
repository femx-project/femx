#pragma once

#include <algorithm>
#include <cstddef>

#include <femx/common/Checks.hpp>
#include <femx/linalg/Backend.hpp>

namespace femx::linalg
{

/** @brief Vector operations associated with one execution backend. */
template <class Backend>
class VectorHandler;

/** @brief Serial CPU vector operations. */
template <>
class VectorHandler<HostCsrBackend> final
{
public:
  explicit VectorHandler(CpuContext& ctx) noexcept
    : ctx_(ctx)
  {
  }

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

  template <class T>
  void copy(VectorView<MemorySpace::Host, T> src,
            VectorView<MemorySpace::Host, T> dst) const
  {
    copy(VectorView<MemorySpace::Host, const T>(src.data(), src.size()),
         dst);
  }

  template <class T>
  void copy(VectorView<MemorySpace::Host, const T> src,
            Vector<MemorySpace::Host, T>&          dst) const
  {
    dst = src;
  }

  template <class T>
  void copy(VectorView<MemorySpace::Host, T> src,
            Vector<MemorySpace::Host, T>&    dst) const
  {
    copy(VectorView<MemorySpace::Host, const T>(src.data(), src.size()),
         dst);
  }

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

  void zero(HostVectorView vals) const;
  void axpby(Real                a,
             HostConstVectorView x,
             Real                b,
             HostVectorView      y) const;
  Real dot(HostConstVectorView x, HostConstVectorView y) const;
  Real squaredNorm(HostConstVectorView x) const;
  void gather(HostConstVectorView src,
              HostConstIndexView  indices,
              HostVectorView      dst) const;
  void scatter(HostConstVectorView src,
               HostConstIndexView  indices,
               HostVectorView      dst) const;

private:
  CpuContext& ctx_;
};

/** @brief CUDA vector operations and explicit Host/Device transfers. */
template <>
class VectorHandler<CudaCsrBackend> final
{
public:
  explicit VectorHandler(CudaContext& ctx) noexcept
    : ctx_(ctx)
  {
  }

  template <class T>
  void copy(const Vector<MemorySpace::Host, T>& src,
            Vector<MemorySpace::Device, T>&     dst) const
  {
    resize(dst, src.size());
    copyStorage(src.data(), MemorySpace::Host, dst.data(), src.size());
  }

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

  template <class T>
  void copy(Vector<MemorySpace::Host, T>&&,
            Vector<MemorySpace::Device, T>&) const = delete;

  template <class T>
  void copy(Vector<MemorySpace::Device, T>&&,
            Vector<MemorySpace::Device, T>&) const = delete;

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

  void copy(DeviceConstVectorView src, DeviceVectorView dst) const;
  void copy(DeviceConstVectorView src, DeviceVector& dst) const;
  void copy(HostConstVectorView src, DeviceVectorView dst) const;
  void copy(HostConstVectorView src, DeviceVector& dst) const;
  void copy(DeviceConstVectorView src, HostVectorView dst) const;
  void copy(DeviceConstVectorView src, HostVector& dst) const;
  void zero(DeviceVectorView vals) const;
  void axpby(Real                  a,
             DeviceConstVectorView x,
             Real                  b,
             DeviceVectorView      y) const;
  void gather(DeviceConstVectorView src,
              DeviceConstIndexView  indices,
              DeviceVectorView      dst) const;
  void scatter(DeviceConstVectorView src,
               DeviceConstIndexView  indices,
               DeviceVectorView      dst) const;
  void dot(DeviceConstVectorView x,
           DeviceConstVectorView y,
           DeviceVectorView      out) const;

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

  CudaContext& ctx_;
};

using HostVectorHandler = VectorHandler<HostCsrBackend>;
using CudaVectorHandler = VectorHandler<CudaCsrBackend>;

} // namespace femx::linalg
