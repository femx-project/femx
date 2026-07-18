#pragma once

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <femx/common/Context.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/VectorView.hpp>

namespace femx
{

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
    if (size < 0)
    {
      throw std::runtime_error("Vector size must be non-negative");
    }
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

template <class T>
class Vector<MemorySpace::Device, T>
{
  static_assert(std::is_trivially_copyable<T>::value,
                "Device vectors require trivially copyable values");

public:
  Vector() = default;

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

  void resize(Index size)
  {
    if (size < 0)
    {
      throw std::runtime_error("Vector size must be non-negative");
    }
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

  void clear() noexcept
  {
    device::release(data_);
    data_ = nullptr;
    size_ = 0;
  }

  Index size() const noexcept
  {
    return size_;
  }

  bool empty() const noexcept
  {
    return size_ == 0;
  }

  T* data() noexcept
  {
    return data_;
  }

  const T* data() const noexcept
  {
    return data_;
  }

  VectorView<MemorySpace::Device, T> view() noexcept
  {
    return {data_, size_};
  }

  VectorView<MemorySpace::Device, const T> view() const noexcept
  {
    return {data_, size_};
  }

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

/** @brief Enqueue an explicit device-to-device clone on context's stream. */
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
