#pragma once

#include <cusparse.h>

#include <cublas_v2.h>

namespace femx::linalg::detail
{

/**
 * @brief Throw when a cuBLAS operation fails.
 *
 * @param[in] status - cuBLAS status to check.
 * @param[in] operation - Operation name included in an error message.
 * @throws std::runtime_error - If `status` does not indicate success.
 */
void checkCublas(cublasStatus_t status, const char* operation);

/**
 * @brief Throw when a cuSPARSE operation fails.
 *
 * @param[in] status - cuSPARSE status to check.
 * @param[in] operation - Operation name included in an error message.
 * @throws std::runtime_error - If `status` does not indicate success.
 */
void checkCusparse(cusparseStatus_t status, const char* operation);

/**
 * @brief Return the shared cuBLAS handle bound to a stream.
 *
 * @param[in] stream - CUDA stream to bind.
 * @return Shared cuBLAS handle.
 * @throws std::runtime_error - If handle creation or stream binding fails.
 */
cublasHandle_t cublasHandle(void* stream);

/**
 * @brief Return the shared cuSPARSE handle bound to a stream.
 *
 * @param[in] stream - CUDA stream to bind.
 * @return Shared cuSPARSE handle.
 * @throws std::runtime_error - If handle creation or stream binding fails.
 */
cusparseHandle_t cusparseHandle(void* stream);

} // namespace femx::linalg::detail
