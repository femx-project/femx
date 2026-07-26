#pragma once

#include <memory>

#include <femx/linalg/Context.hpp>
#include <femx/linalg/cuda/CudaVectorHandler.hpp>

namespace femx::linalg
{

namespace detail
{
class CudaHandles;
struct CudaContextAccess;
} // namespace detail

/** @brief Own CUDA execution resources for one Device pipeline. */
class CudaContext final : public Context<MemorySpace::Device>
{
public:
  /** @brief Create a context owning one non-blocking CUDA stream. */
  CudaContext();

  /** @brief Destroy CUDA resources after queued work completes. */
  ~CudaContext() override;

  CudaContext(const CudaContext&)            = delete;
  CudaContext& operator=(const CudaContext&) = delete;
  CudaContext(CudaContext&&)                 = delete;
  CudaContext& operator=(CudaContext&&)      = delete;

  /** @brief Return the owned CUDA vector operations. */
  CudaVectorHandler& vectors() noexcept override;

  /**
   * @brief Return the full Device element range.
   *
   * @param[in] count - Element count.
   * @return Full half-open element range.
   * @throws std::runtime_error - If `count` is negative.
   */
  IndexRange elementRange(Index count) const override;

  /** @brief Wait for all work queued on this context. */
  void sync() const override;

  /** @brief Return the opaque native CUDA stream handle. */
  void* stream() const noexcept;

  /** @brief Return whether a usable CUDA device is available. */
  static bool available() noexcept;

private:
  friend struct detail::CudaContextAccess;

  void*                                stream_{nullptr}; ///< Owned CUDA stream.
  std::unique_ptr<detail::CudaHandles> handles_;         ///< Owned CUDA library handles.
  std::shared_ptr<void>                sparse_state_;    ///< Cached sparse operation state.
  CudaVectorHandler                    vectors_;         ///< Owned CUDA vector operations.
};

} // namespace femx::linalg
