#pragma once

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <femx/common/Checks.hpp>
#include <femx/common/Context.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/View.hpp>

namespace femx
{

/** @brief Mutable non-owning view of contiguous Device real values. */
using DeviceVectorView      = VectorView<MemorySpace::Device, Real>;
/** @brief Read-only non-owning view of contiguous Device real values. */
using DeviceConstVectorView = VectorView<MemorySpace::Device, const Real>;
/** @brief Read-only non-owning view of Host indices. */
using HostConstIndexView    = VectorView<MemorySpace::Host, const Index>;
/** @brief Read-only non-owning view of Device indices. */
using DeviceConstIndexView  = VectorView<MemorySpace::Device, const Index>;

/**
 * @brief Owning contiguous host vector with the femx signed index type.
 *
 * Its container interface follows `std::vector`; resizing value-initializes
 * all entries and host views remain valid only until storage is reallocated.
 */
template <class T>
class Vector<MemorySpace::Host, T>
{
public:
  Vector() = default;

  Vector(std::initializer_list<T> vals)
    : vals_(vals)
  {
  }

  explicit Vector(Index size)
    : vals_(checkedSize(size), T{})
  {
  }

  Vector(Index size, const T& val)
    : vals_(checkedSize(size), val)
  {
  }

  template <class U>
  Vector(VectorView<MemorySpace::Host, U> view)
  {
    assign(view);
  }

  Vector(const Vector&)     = default;
  Vector(Vector&&) noexcept = default;

  Vector& operator=(const Vector&)     = default;
  Vector& operator=(Vector&&) noexcept = default;

  Vector& operator=(std::initializer_list<T> vals)
  {
    vals_.assign(vals.begin(), vals.end());
    return *this;
  }

  template <class U>
  Vector& operator=(VectorView<MemorySpace::Host, U> view)
  {
    assign(view);
    return *this;
  }

  void resize(Index size)
  {
    vals_.assign(checkedSize(size), T{});
  }

  void assign(Index size, const T& val)
  {
    vals_.assign(checkedSize(size), val);
  }

  Index size() const noexcept
  {
    return static_cast<Index>(vals_.size());
  }

  bool empty() const noexcept
  {
    return vals_.empty();
  }

  void clear() noexcept
  {
    vals_.clear();
  }

  void reserve(Index size)
  {
    vals_.reserve(checkedSize(size));
  }

  void push_back(const T& val)
  {
    vals_.push_back(val);
  }

  void push_back(T&& val)
  {
    vals_.push_back(std::move(val));
  }

  template <class... Args>
  T& emplace_back(Args&&... args)
  {
    return vals_.emplace_back(std::forward<Args>(args)...);
  }

  T& front()
  {
    return vals_.front();
  }

  const T& front() const
  {
    return vals_.front();
  }

  T& back()
  {
    return vals_.back();
  }

  const T& back() const
  {
    return vals_.back();
  }

  T& operator[](Index i)
  {
    return vals_[static_cast<std::size_t>(i)];
  }

  const T& operator[](Index i) const
  {
    return vals_[static_cast<std::size_t>(i)];
  }

  T* data() noexcept
  {
    return vals_.data();
  }

  const T* data() const noexcept
  {
    return vals_.data();
  }

  T* begin() noexcept
  {
    return vals_.data();
  }

  const T* begin() const noexcept
  {
    return vals_.data();
  }

  T* end() noexcept
  {
    return vals_.data() + vals_.size();
  }

  const T* end() const noexcept
  {
    return vals_.data() + vals_.size();
  }

  VectorView<MemorySpace::Host, T> view() noexcept
  {
    return {data(), size()};
  }

  VectorView<MemorySpace::Host, const T> view() const noexcept
  {
    return {data(), size()};
  }

  void setZero()
  {
    std::fill(vals_.begin(), vals_.end(), T{});
  }

private:
  static std::size_t checkedSize(Index size)
  {
    require(size >= 0, "Vector size must be non-negative");
    return static_cast<std::size_t>(size);
  }

  template <class U>
  void assign(VectorView<MemorySpace::Host, U> view)
  {
    vals_.resize(checkedSize(view.size()));
    for (Index i = 0; i < view.size(); ++i)
    {
      vals_[static_cast<std::size_t>(i)] = view[i];
    }
  }

  std::vector<T> vals_;
};

/**
 * @brief Move-only owner of a contiguous CUDA device allocation.
 *
 * Resizing replaces the allocation and invalidates all views and borrowed
 * pointers. Host access requires an explicit `copy()` operation.
 */
template <class T>
class Vector<MemorySpace::Device, T>
{
  static_assert(std::is_trivially_copyable<T>::value,
                "Device vectors require trivially copyable values");

public:
  Vector() = default;

  /** @brief Allocate and zero `size` device values. */
  explicit Vector(Index size)
  {
    resize(size);
  }

  ~Vector()
  {
    clear();
  }

  Vector(const Vector&) = delete;

  Vector(Vector&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0))
  {
  }

  Vector& operator=(const Vector&) = delete;

  Vector& operator=(Vector&& other) noexcept
  {
    if (this != &other)
    {
      clear();
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  /** @brief Replace storage with a zeroed allocation of `size` values. */
  void resize(Index size)
  {
    require(size >= 0, "Vector size must be non-negative");
    T* replacement = nullptr;
    if (size > 0)
    {
      replacement = static_cast<T*>(device::allocate(bytesFor(size)));
      try
      {
        device::zero(replacement, bytesFor(size));
        // The context-free memset runs on the default stream. CudaContext uses
        // a non-blocking stream, so make initialization complete before the
        // replacement pointer becomes visible to work on another stream.
        device::synchronize(nullptr);
      }
      catch (...)
      {
        device::release(replacement);
        throw;
      }
    }
    device::release(data_);
    data_ = replacement;
    size_ = size;
  }

  /** @brief Release device storage and reset the size to zero. */
  void clear() noexcept
  {
    device::release(data_);
    data_ = nullptr;
    size_ = 0;
  }

  /** @brief Return the number of allocated values. */
  Index size() const noexcept
  {
    return size_;
  }

  /** @brief Return whether no values are allocated. */
  bool empty() const noexcept
  {
    return size_ == 0;
  }

  /** @brief Return the mutable device pointer. */
  T* data() noexcept
  {
    return data_;
  }

  /** @brief Return the device pointer. */
  const T* data() const noexcept
  {
    return data_;
  }

  /** @brief Return a mutable non-owning device view. */
  VectorView<MemorySpace::Device, T> view() noexcept
  {
    return {data_, size_};
  }

  /** @brief Return a read-only non-owning device view. */
  VectorView<MemorySpace::Device, const T> view() const noexcept
  {
    return {data_, size_};
  }

  /** @brief Enqueue zeroing of all values on `ctx`. */
  void setZero(CudaContext& ctx)
  {
    if (data_ != nullptr)
    {
      device::zero(data_, bytes(), ctx.stream());
    }
  }

private:
  std::size_t bytes() const noexcept
  {
    return bytesFor(size_);
  }

  static std::size_t bytesFor(Index size) noexcept
  {
    return static_cast<std::size_t>(size) * sizeof(T);
  }

  T*    data_{nullptr};
  Index size_{0};
};

/** @brief Copy equal-sized Host views without changing storage. */
inline void copy(HostConstVectorView src,
                 HostVectorView      dst,
                 CpuContext&)
{
  require(src.size() == dst.size(),
          "Host view copy requires equal sizes");
  if (src.empty() || src.data() == dst.data())
  {
    return;
  }
  require(!detail::overlaps(src, dst),
          "Host view copy does not support partial overlap");
  std::copy(src.begin(), src.end(), dst.begin());
}

/** @brief Resize and copy into owning Host storage. */
inline void copy(HostConstVectorView src,
                 HostVector&         dst,
                 CpuContext&)
{
  dst = src;
}

/** @brief Set every entry of a Host view to zero. */
inline void zero(HostVectorView vals, CpuContext&)
{
  std::fill(vals.begin(), vals.end(), Real{});
}

/** @brief Replace a Host view by `a * x + b * y`. */
inline void axpby(Real                a,
                  HostConstVectorView x,
                  Real                b,
                  HostVectorView      y,
                  CpuContext&)
{
  require(x.size() == y.size(),
          "Host axpby requires equal vector sizes");
  require(x.data() == y.data() || !detail::overlaps(x, y),
          "Host axpby does not support partial overlap");
  for (Index i = 0; i < x.size(); ++i)
  {
    y[i] = a * x[i] + b * y[i];
  }
}

/** @brief Set `dst[i] = src[indices[i]]` on Host. */
inline void gather(HostConstVectorView src,
                   HostConstIndexView  indices,
                   HostVectorView      dst,
                   CpuContext&)
{
  require(indices.size() == dst.size(),
          "Host gather output size mismatch");
  require(!detail::overlaps(src, dst),
          "Host gather does not support aliased vectors");
  for (Index i = 0; i < indices.size(); ++i)
  {
    require(indices[i] >= 0 && indices[i] < src.size(),
            "Host gather index is out of range");
    dst[i] = src[indices[i]];
  }
}

/** @brief Set `dst[indices[i]] = src[i]` on Host. */
inline void scatter(HostConstVectorView src,
                    HostConstIndexView  indices,
                    HostVectorView      dst,
                    CpuContext&)
{
  require(src.size() == indices.size(),
          "Host scatter input size mismatch");
  require(!detail::overlaps(src, dst),
          "Host scatter does not support aliased vectors");
  for (Index i = 0; i < indices.size(); ++i)
  {
    require(indices[i] >= 0 && indices[i] < dst.size(),
            "Host scatter index is out of range");
    dst[indices[i]] = src[i];
  }
}

/**
 * @brief Enqueue a same-sized device-view copy without changing storage.
 * @param src Read-only source view.
 * @param dst Destination view; it must not partially overlap `src`.
 * @param ctx CUDA stream on which the copy is enqueued.
 */
void copy(DeviceConstVectorView src,
          DeviceVectorView      dst,
          CudaContext&          ctx);

/** @brief Resize and enqueue a copy into owning Device storage. */
inline void copy(DeviceConstVectorView src,
                 DeviceVector&         dst,
                 CudaContext&          ctx)
{
  if (dst.size() != src.size())
  {
    dst.resize(src.size());
  }
  copy(src, dst.view(), ctx);
}

/** @brief Enqueue zeroing of a Device view. */
void zero(DeviceVectorView vals, CudaContext& ctx);

/**
 * @brief Replace `y` by `a * x + b * y` without changing storage.
 * @param a Scale applied to `x`.
 * @param x Read-only input view.
 * @param b Scale applied to the previous values of `y`.
 * @param y Input/output view with the same size as `x`.
 * @param ctx CUDA stream on which the operation is enqueued.
 */
void axpby(Real                  a,
           DeviceConstVectorView x,
           Real                  b,
           DeviceVectorView      y,
           CudaContext&          ctx);

/**
 * @brief Enqueue `dst[i] = src[indices[i]]` using cuSPARSE.
 *
 * Indices must be distinct and in `[0, src.size())`.
 */
void gather(DeviceConstVectorView src,
            DeviceConstIndexView  indices,
            DeviceVectorView      dst,
            CudaContext&          ctx);

/**
 * @brief Enqueue `dst[indices[i]] = src[i]` using cuSPARSE.
 *
 * Indices must be distinct and in `[0, dst.size())`; unindexed destination
 * entries are left unchanged.
 */
void scatter(DeviceConstVectorView src,
             DeviceConstIndexView  indices,
             DeviceVectorView      dst,
             CudaContext&          ctx);

/** @brief Overwrite the one-entry Device `out` with `x^T y` via cuBLAS. */
void dot(DeviceConstVectorView x,
         DeviceConstVectorView y,
         DeviceVectorView      out,
         CudaContext&          ctx);

#if !defined(FEMX_HAS_CUDA)
namespace detail
{
[[noreturn]] inline void throwCudaVectorUnavailable()
{
  throw std::runtime_error(
      "femx was built without the CUDA execution backend");
}
} // namespace detail

inline void copy(DeviceConstVectorView,
                 DeviceVectorView,
                 CudaContext&)
{
  detail::throwCudaVectorUnavailable();
}

inline void zero(DeviceVectorView, CudaContext&)
{
  detail::throwCudaVectorUnavailable();
}

inline void axpby(Real,
                  DeviceConstVectorView,
                  Real,
                  DeviceVectorView,
                  CudaContext&)
{
  detail::throwCudaVectorUnavailable();
}

inline void gather(DeviceConstVectorView,
                   DeviceConstIndexView,
                   DeviceVectorView,
                   CudaContext&)
{
  detail::throwCudaVectorUnavailable();
}

inline void scatter(DeviceConstVectorView,
                    DeviceConstIndexView,
                    DeviceVectorView,
                    CudaContext&)
{
  detail::throwCudaVectorUnavailable();
}

inline void dot(DeviceConstVectorView,
                DeviceConstVectorView,
                DeviceVectorView,
                CudaContext&)
{
  detail::throwCudaVectorUnavailable();
}
#endif

/** @brief Return the squared Euclidean norm of a Host vector. */
inline Real squaredNorm(HostConstVectorView x, CpuContext&)
{
  Real val = 0.0;
  for (Index i = 0; i < x.size(); ++i)
  {
    val += x[i] * x[i];
  }
  return val;
}

/**
 * @brief Enqueue an explicit host-to-device copy on context's stream.
 *
 * The source and destination storage must remain alive until the context is
 * synchronized or later work on the same stream has consumed the copy.
 */
template <class T>
void copy(const Vector<MemorySpace::Host, T>& src,
          Vector<MemorySpace::Device, T>&     dst,
          CudaContext&                        ctx)
{
  if (dst.size() != src.size())
  {
    dst.resize(src.size());
  }
  if (!src.empty())
  {
    device::copy(dst.data(),
                 MemorySpace::Device,
                 src.data(),
                 MemorySpace::Host,
                 static_cast<std::size_t>(src.size()) * sizeof(T),
                 ctx.stream());
  }
}

template <class T>
void copy(Vector<MemorySpace::Host, T>&&,
          Vector<MemorySpace::Device, T>&,
          CudaContext&) = delete;

/**
 * @brief Enqueue an explicit device-to-device clone on context's stream.
 *
 * Source and destination storage must remain alive until the queued copy has
 * completed or later work on the same stream has consumed it.
 */
template <class T>
void copy(const Vector<MemorySpace::Device, T>& src,
          Vector<MemorySpace::Device, T>&       dst,
          CudaContext&                          ctx)
{
  if (&src == &dst)
  {
    return;
  }
  if (dst.size() != src.size())
  {
    dst.resize(src.size());
  }
  if (!src.empty())
  {
    device::copy(dst.data(),
                 MemorySpace::Device,
                 src.data(),
                 MemorySpace::Device,
                 static_cast<std::size_t>(src.size()) * sizeof(T),
                 ctx.stream());
  }
}

template <class T>
void copy(Vector<MemorySpace::Device, T>&&,
          Vector<MemorySpace::Device, T>&,
          CudaContext&) = delete;

/**
 * @brief Enqueue an explicit device-to-host copy on context's stream.
 *
 * Do not consume or release either storage until the context is synchronized.
 */
template <class T>
void copy(const Vector<MemorySpace::Device, T>& src,
          Vector<MemorySpace::Host, T>&         dst,
          CudaContext&                          ctx)
{
  if (dst.size() != src.size())
  {
    dst.resize(src.size());
  }
  if (!src.empty())
  {
    device::copy(dst.data(),
                 MemorySpace::Host,
                 src.data(),
                 MemorySpace::Device,
                 static_cast<std::size_t>(src.size()) * sizeof(T),
                 ctx.stream());
  }
}

template <class T>
/** @brief Resize a host vector if needed, otherwise set its values to zero. */
void resizeOrZero(Vector<MemorySpace::Host, T>& out, Index size)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    out.setZero();
  }
}

template <class T>
/** @brief Resize a device vector if needed, otherwise enqueue zeroing. */
void resizeOrZero(Vector<MemorySpace::Device, T>& out,
                  Index                           size,
                  CudaContext&                    ctx)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    out.setZero(ctx);
  }
}

} // namespace femx
