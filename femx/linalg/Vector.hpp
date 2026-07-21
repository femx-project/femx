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
 * @brief Own a contiguous Host vector with the femx signed index type.
 *
 * Its container interface follows `std::vector`; resizing value-initializes
 * all entries and host views remain valid only until storage is reallocated.
 */
template <class T>
class Vector<MemorySpace::Host, T>
{
public:
  Vector() = default;

  /**
   * @brief Construct a vector from initializer-list values.
   *
   * @param[in] vals - Values to copy.
   */
  Vector(std::initializer_list<T> vals)
    : vals_(vals)
  {
  }

  /**
   * @brief Construct a value-initialized vector.
   *
   * @param[in] size - Number of values.
   * @throws std::runtime_error - If `size` is negative.
   */
  explicit Vector(Index size)
    : vals_(checkedSize(size), T{})
  {
  }

  /**
   * @brief Construct a vector filled with one value.
   *
   * @param[in] size - Number of values.
   * @param[in] val - Initial value for every entry.
   * @throws std::runtime_error - If `size` is negative.
   */
  Vector(Index size, const T& val)
    : vals_(checkedSize(size), val)
  {
  }

  /**
   * @brief Construct a vector by copying a Host view.
   *
   * @param[in] view - Values to copy.
   * @throws std::runtime_error - If the view size is negative.
   */
  template <class U>
  Vector(VectorView<MemorySpace::Host, U> view)
  {
    assign(view);
  }

  Vector(const Vector&) = default;

  Vector(Vector&&) noexcept = default;

  Vector& operator=(const Vector&) = default;

  Vector& operator=(Vector&&) noexcept = default;

  /**
   * @brief Replace the vector with initializer-list values.
   *
   * @param[in] vals - Values to copy.
   * @return This vector.
   */
  Vector& operator=(std::initializer_list<T> vals)
  {
    vals_.assign(vals.begin(), vals.end());
    return *this;
  }

  /**
   * @brief Replace the vector by copying a Host view.
   *
   * @param[in] view - Values to copy.
   * @return This vector.
   * @throws std::runtime_error - If the view size is negative.
   */
  template <class U>
  Vector& operator=(VectorView<MemorySpace::Host, U> view)
  {
    assign(view);
    return *this;
  }

  /**
   * @brief Replace storage with value-initialized entries.
   *
   * @param[in] size - New number of entries.
   * @throws std::runtime_error - If `size` is negative.
   */
  void resize(Index size)
  {
    vals_.assign(checkedSize(size), T{});
  }

  /**
   * @brief Replace storage with copies of one value.
   *
   * @param[in] size - New number of entries.
   * @param[in] val - Value assigned to every entry.
   * @throws std::runtime_error - If `size` is negative.
   */
  void assign(Index size, const T& val)
  {
    vals_.assign(checkedSize(size), val);
  }

  /** @brief Return the number of stored values. */
  Index size() const noexcept
  {
    return static_cast<Index>(vals_.size());
  }

  /** @brief Report whether the vector is empty. */
  bool empty() const noexcept
  {
    return vals_.empty();
  }

  /** @brief Remove all values. */
  void clear() noexcept
  {
    vals_.clear();
  }

  /**
   * @brief Reserve storage for at least the requested number of values.
   *
   * @param[in] size - Requested capacity.
   * @throws std::runtime_error - If `size` is negative.
   */
  void reserve(Index size)
  {
    vals_.reserve(checkedSize(size));
  }

  /**
   * @brief Append a copied value.
   *
   * @param[in] val - Value to append.
   */
  void push_back(const T& val)
  {
    vals_.push_back(val);
  }

  /**
   * @brief Append a moved value.
   *
   * @param[in] val - Value to move into the vector.
   */
  void push_back(T&& val)
  {
    vals_.push_back(std::move(val));
  }

  /**
   * @brief Construct a value at the end of the vector.
   *
   * @param args - Arguments forwarded to the value constructor.
   * @return Reference to the appended value.
   */
  template <class... Args>
  T& emplace_back(Args&&... args)
  {
    return vals_.emplace_back(std::forward<Args>(args)...);
  }

  /** @brief Access the first value. */
  T& front()
  {
    return vals_.front();
  }

  /** @brief Access the first value. */
  const T& front() const
  {
    return vals_.front();
  }

  /** @brief Access the last value. */
  T& back()
  {
    return vals_.back();
  }

  /** @brief Access the last value. */
  const T& back() const
  {
    return vals_.back();
  }

  /**
   * @brief Access a value without bounds checking.
   *
   * @param[in] i - Value index.
   * @return Reference to the indexed value.
   */
  T& operator[](Index i)
  {
    return vals_[static_cast<std::size_t>(i)];
  }

  /**
   * @brief Access a value without bounds checking.
   *
   * @param[in] i - Value index.
   * @return Read-only reference to the indexed value.
   */
  const T& operator[](Index i) const
  {
    return vals_[static_cast<std::size_t>(i)];
  }

  /** @brief Return the address of the first stored value. */
  T* data() noexcept
  {
    return vals_.data();
  }

  /** @brief Return the address of the first stored value. */
  const T* data() const noexcept
  {
    return vals_.data();
  }

  /** @brief Return an iterator to the first value. */
  T* begin() noexcept
  {
    return vals_.data();
  }

  /** @brief Return a read-only iterator to the first value. */
  const T* begin() const noexcept
  {
    return vals_.data();
  }

  /** @brief Return an iterator past the last value. */
  T* end() noexcept
  {
    return vals_.data() + vals_.size();
  }

  /** @brief Return a read-only iterator past the last value. */
  const T* end() const noexcept
  {
    return vals_.data() + vals_.size();
  }

  /** @brief Return a mutable view of the stored values. */
  VectorView<MemorySpace::Host, T> view() noexcept
  {
    return {data(), size()};
  }

  /** @brief Return a read-only view of the stored values. */
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

  std::vector<T> vals_; ///< Owned Host values.
};

/**
 * @brief Own a move-only contiguous CUDA Device allocation.
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

  /**
   * @brief Construct a zeroed Device vector.
   *
   * @param[in] size - Number of values.
   * @throws std::runtime_error - If `size` is negative or a CUDA operation fails.
   */
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

  /**
   * @brief Replace storage with a zeroed Device allocation.
   *
   * @param[in] size - New number of values.
   * @throws std::runtime_error - If `size` is negative or a CUDA operation fails.
   */
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

  /** @brief Release the Device allocation. */
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

  /** @brief Report whether the vector is empty. */
  bool empty() const noexcept
  {
    return size_ == 0;
  }

  /** @brief Return the Device address of the first value. */
  T* data() noexcept
  {
    return data_;
  }

  /** @brief Return the Device address of the first value. */
  const T* data() const noexcept
  {
    return data_;
  }

  /** @brief Return a mutable Device view of the allocation. */
  VectorView<MemorySpace::Device, T> view() noexcept
  {
    return {data_, size_};
  }

  /** @brief Return a read-only Device view of the allocation. */
  VectorView<MemorySpace::Device, const T> view() const noexcept
  {
    return {data_, size_};
  }

private:
  static std::size_t bytesFor(Index size) noexcept
  {
    return static_cast<std::size_t>(size) * sizeof(T);
  }

  T*    data_{nullptr}; ///< Owned Device allocation.
  Index size_{0};       ///< Number of allocated values.
};

} // namespace femx
