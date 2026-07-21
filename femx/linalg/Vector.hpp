#pragma once

#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

#include <femx/common/Checks.hpp>
#include <femx/common/Cuda.hpp>
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
 * pointers. Host access requires an explicit `CudaVectorHandler::copy()`
 * operation.
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
      replacement = static_cast<T*>(cuda::allocate(bytesFor(size)));
      try
      {
        cuda::zero(replacement, bytesFor(size));
        // The context-free memset runs on the default stream. CudaContext uses
        // a non-blocking stream, so make initialization complete before the
        // replacement pointer becomes visible to work on another stream.
        cuda::sync(nullptr);
      }
      catch (...)
      {
        cuda::release(replacement);
        throw;
      }
    }
    cuda::release(data_);
    data_ = replacement;
    size_ = size;
  }

  /** @brief Release device storage and reset the size to zero. */
  void clear() noexcept
  {
    cuda::release(data_);
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

private:
  static std::size_t bytesFor(Index size) noexcept
  {
    return static_cast<std::size_t>(size) * sizeof(T);
  }

  T*    data_{nullptr};
  Index size_{0};
};

} // namespace femx
