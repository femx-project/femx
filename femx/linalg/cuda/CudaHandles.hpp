#pragma once

#include <memory>

#if defined(FEMX_HAS_CUDA)
#include <cusparse.h>

#include <cublas_v2.h>
#endif

namespace femx::linalg
{

class CudaContext;

namespace detail
{

/**
 * @brief Own cuBLAS and cuSPARSE handles bound to one CUDA stream.
 */
class CudaHandles final
{
public:
  /**
   * @brief Create CUDA library handles bound to a stream.
   *
   * @param[in] stream - CUDA stream used by both handles.
   * @throws - If handle creation or stream binding fails.
   */
  explicit CudaHandles(void* stream);

  /**
   * @brief Destroy the owned CUDA library handles.
   */
  ~CudaHandles();

  CudaHandles(const CudaHandles&)            = delete;
  CudaHandles& operator=(const CudaHandles&) = delete;

#if defined(FEMX_HAS_CUDA)

  /**
   * @brief Return the owned cuBLAS handle.
   */
  cublasHandle_t cublas() const noexcept;

  /**
   * @brief Return the owned cuSPARSE handle.
   */
  cusparseHandle_t cusparse() const noexcept;

#endif

private:
  void* cublas_{nullptr};   ///< Owned cuBLAS handle.
  void* cusparse_{nullptr}; ///< Owned cuSPARSE handle.
};

/**
 * @brief Create CUDA library handles for a stream.
 *
 * @param[in] stream - CUDA stream used by the handles.
 * @return Owned CUDA library handles.
 */
std::unique_ptr<CudaHandles> makeCudaHandles(void* stream);

/**
 * @brief Return context-owned sparse operation state.
 *
 * @param[in] ctx - CUDA context owning the state.
 * @return Mutable opaque sparse state storage.
 */
std::shared_ptr<void>& cudaSparseState(CudaContext& ctx);

#if defined(FEMX_HAS_CUDA)
/**
 * @brief Throw when a cuBLAS operation fails.
 *
 * @param[in] status - cuBLAS status to check.
 * @param[in] operation - Operation name included in an error message.
 * @throws - If `status` does not indicate success.
 */
void checkCublas(cublasStatus_t status, const char* operation);

/**
 * @brief Throw when a cuSPARSE operation fails.
 *
 * @param[in] status - cuSPARSE status to check.
 * @param[in] operation - Operation name included in an error message.
 * @throws - If `status` does not indicate success.
 */
void checkCusparse(cusparseStatus_t status, const char* operation);

/**
 * @brief Return the context-owned cuBLAS handle.
 *
 * @param[in] ctx - CUDA context owning the handle.
 * @return Context-owned cuBLAS handle.
 */
cublasHandle_t cublasHandle(CudaContext& ctx);

/**
 * @brief Return the context-owned cuSPARSE handle.
 *
 * @param[in] ctx - CUDA context owning the handle.
 * @return Context-owned cuSPARSE handle.
 */
cusparseHandle_t cusparseHandle(CudaContext& ctx);
#endif

} // namespace detail
} // namespace femx::linalg
