#pragma once

#include <memory>

#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>

namespace femx::linalg
{

/** @brief Provide matrix operations for an execution backend. */
template <class Backend>
class MatrixHandler;

/**
 * @brief Provide serial CPU sparse and dense matrix operations.
 *
 * Explicit CSR transposes cache their derived pattern and value permutation in
 * this handler so repeated transposes of one layout update only numeric values.
 */
template <>
class MatrixHandler<HostCsrBackend> final
{
public:
  /**
   * @brief Bind matrix operations to a CPU context.
   *
   * @param[in] ctx - CPU execution context.
   */
  explicit MatrixHandler(CpuContext& ctx) noexcept
    : ctx_(ctx), vec_handler_(ctx)
  {
  }

  /**
   * @brief Set every numeric value in a CSR matrix to zero.
   *
   * @param[in,out] mat - Matrix whose values are cleared.
   */
  void zero(HostCsrMatrix& mat) const;

  /**
   * @brief Set every value in a dense matrix to zero.
   *
   * @param[in,out] mat - Matrix whose values are cleared.
   */
  void zero(DenseMatrix& mat) const;

  /**
   * @brief Copy CSR values between matrices with the same layout.
   *
   * @param[in] src - Source matrix.
   * @param[out] dst - Destination matrix.
   * @throws std::runtime_error - If the matrix layouts are incompatible.
   */
  void copy(const HostCsrMatrix& src, HostCsrMatrix& dst) const;

  /**
   * @brief Construct or update the explicit CSR transpose of `src`.
   *
   * A changed source layout replaces `dst`; an unchanged layout preserves its
   * allocations and updates only values. In-place transpose is not supported.
   *
   * @param[in] src - Source matrix.
   * @param[in,out] dst - Destination replaced or updated with the transpose.
   * @throws std::runtime_error - If source and destination are the same matrix
   * or the source structure is invalid.
   */
  void transpose(const HostCsrMatrix& src, HostCsrMatrix& dst) const;

  /**
   * @brief Compute `y = alpha * mat * x + beta * y` for a CSR matrix.
   *
   * @param[in] mat - CSR matrix.
   * @param[in] x - Input vector.
   * @param[in,out] y - Output vector, scaled by `beta` before accumulation.
   * @param[in] alpha - Matrix-product scale.
   * @param[in] beta - Existing-output scale.
   * @throws std::runtime_error - If dimensions or storage overlap are invalid.
   */
  void matvec(const HostCsrMatrix& mat,
              HostConstVectorView  x,
              HostVectorView       y,
              Real                 alpha = 1.0,
              Real                 beta  = 0.0) const;

  /**
   * @brief Compute `y = alpha * mat^T * x + beta * y` for a CSR matrix.
   *
   * @param[in] mat - CSR matrix.
   * @param[in] x - Input vector.
   * @param[in,out] y - Output vector, scaled by `beta` before accumulation.
   * @param[in] alpha - Matrix-product scale.
   * @param[in] beta - Existing-output scale.
   * @throws std::runtime_error - If dimensions or storage overlap are invalid.
   */
  void matvecT(const HostCsrMatrix& mat,
               HostConstVectorView  x,
               HostVectorView       y,
               Real                 alpha = 1.0,
               Real                 beta  = 0.0) const;

  /**
   * @brief Resize the output if needed and compute `out = mat * x`.
   *
   * @param[in] mat - CSR matrix.
   * @param[in] x - Input vector.
   * @param[out] out - Resized output vector.
   * @throws std::runtime_error - If dimensions or storage overlap are invalid.
   */
  void matvec(const HostCsrMatrix& mat,
              HostConstVectorView  x,
              HostVector&          out) const;

  /**
   * @brief Resize the output if needed and compute `out = mat^T * x`.
   *
   * @param[in] mat - CSR matrix.
   * @param[in] x - Input vector.
   * @param[out] out - Resized output vector.
   * @throws std::runtime_error - If dimensions or storage overlap are invalid.
   */
  void matvecT(const HostCsrMatrix& mat,
               HostConstVectorView  x,
               HostVector&          out) const;

  /**
   * @brief Compute `y = alpha * mat * x + beta * y` for a dense view.
   *
   * @param[in] mat - Dense matrix view.
   * @param[in] x - Input vector.
   * @param[in,out] y - Output vector, scaled by `beta` before accumulation.
   * @param[in] alpha - Matrix-product scale.
   * @param[in] beta - Existing-output scale.
   * @throws std::runtime_error - If dimensions or storage overlap are invalid.
   */
  void matvec(HostMatrixView<const Real> mat,
              HostConstVectorView        x,
              HostVectorView             y,
              Real                       alpha = 1.0,
              Real                       beta  = 0.0) const;

  /**
   * @brief Compute `y = alpha * mat^T * x + beta * y` for a dense view.
   *
   * @param[in] mat - Dense matrix view.
   * @param[in] x - Input vector.
   * @param[in,out] y - Output vector, scaled by `beta` before accumulation.
   * @param[in] alpha - Matrix-product scale.
   * @param[in] beta - Existing-output scale.
   * @throws std::runtime_error - If dimensions or storage overlap are invalid.
   */
  void matvecT(HostMatrixView<const Real> mat,
               HostConstVectorView        x,
               HostVectorView             y,
               Real                       alpha = 1.0,
               Real                       beta  = 0.0) const;

  /**
   * @brief Leave a Host CSR matrix unchanged after assembly.
   *
   * @param[in,out] mat - Assembled matrix.
   */
  void finalize(HostCsrMatrix& mat) const noexcept
  {
    static_cast<void>(mat);
  }

private:
  static void checkCopy(const HostCsrMatrix& src, const HostCsrMatrix& dst)
  {
    require(src.rows() == dst.rows() && src.cols() == dst.cols()
                && src.nnz() == dst.nnz()
                && src.pattern().layoutId() == dst.pattern().layoutId()
                && src.vals().size() == src.nnz()
                && dst.vals().size() == dst.nnz(),
            "CsrMatrix copy requires compatible source and destination graphs");
  }

  CpuContext&                   ctx_;             ///< Bound CPU context.
  HostVectorHandler             vec_handler_;     ///< Host vector operations.
  mutable std::shared_ptr<void> transpose_state_; ///< Cached transpose state.
};

/**
 * @brief Provide CUDA sparse and dense matrix operations.
 *
 * Operations are enqueued on the stream owned by the bound context. Explicit
 * CSR transposes cache their derived pattern, value permutation, and workspace
 * by source layout in that context. Synchronize the context before reading
 * Host outputs from an asynchronous copy.
 */
template <>
class MatrixHandler<CudaCsrBackend> final
{
public:
  /**
   * @brief Bind matrix operations to a CUDA context.
   *
   * @param[in] ctx - CUDA execution context.
   */
  explicit MatrixHandler(CudaContext& ctx) noexcept
    : ctx_(ctx), vec_handler_(ctx)
  {
  }

  /**
   * @brief Enqueue setting every numeric value in a CSR matrix to zero.
   *
   * @param[in,out] mat - Device matrix whose values are cleared.
   * @throws std::runtime_error - If a CUDA operation fails.
   */
  void zero(DeviceCsrMatrix& mat) const;

  /**
   * @brief Enqueue copying CSR values from a same-layout Host matrix.
   *
   * @param[in] src - Source Host matrix.
   * @param[out] dst - Destination Device matrix.
   * @throws std::runtime_error - If layouts are incompatible or a CUDA
   * operation fails.
   */
  void copy(const HostCsrMatrix& src, DeviceCsrMatrix& dst) const;

  /**
   * @brief Enqueue copying CSR values between same-layout Device matrices.
   *
   * @param[in] src - Source Device matrix.
   * @param[out] dst - Destination Device matrix.
   * @throws std::runtime_error - If layouts are incompatible or a CUDA
   * operation fails.
   */
  void copy(const DeviceCsrMatrix& src, DeviceCsrMatrix& dst) const;

  /**
   * @brief Enqueue copying CSR values to a same-layout Host matrix.
   *
   * @param[in] src - Source Device matrix.
   * @param[out] dst - Destination Host matrix.
   * @throws std::runtime_error - If layouts are incompatible or a CUDA
   * operation fails.
   */
  void copy(const DeviceCsrMatrix& src, HostCsrMatrix& dst) const;

  /**
   * @brief Enqueue construction or update of the explicit CSR transpose.
   *
   * A changed source layout replaces `dst`; an unchanged layout preserves its
   * allocations and updates only values. In-place transpose is not supported.
   *
   * @param[in] src - Source Device matrix.
   * @param[in,out] dst - Device destination replaced or updated with the
   * transpose.
   * @throws std::runtime_error - If source and destination alias, the source
   * structure is invalid, or a CUDA operation fails.
   */
  void transpose(const DeviceCsrMatrix& src, DeviceCsrMatrix& dst) const;

  /**
   * @brief Enqueue `y = alpha * mat * x + beta * y` for a CSR matrix.
   *
   * @param[in] mat - Device CSR matrix.
   * @param[in] x - Device input vector.
   * @param[in,out] y - Device output scaled by `beta` before accumulation.
   * @param[in] alpha - Matrix-product scale.
   * @param[in] beta - Existing-output scale.
   * @throws std::runtime_error - If inputs are invalid or a CUDA operation
   * fails.
   */
  void matvec(const DeviceCsrMatrix& mat,
              DeviceConstVectorView  x,
              DeviceVectorView       y,
              Real                   alpha = 1.0,
              Real                   beta  = 0.0) const;

  /**
   * @brief Enqueue `y = alpha * mat^T * x + beta * y` for a CSR matrix.
   *
   * @param[in] mat - Device CSR matrix.
   * @param[in] x - Device input vector.
   * @param[in,out] y - Device output scaled by `beta` before accumulation.
   * @param[in] alpha - Matrix-product scale.
   * @param[in] beta - Existing-output scale.
   * @throws std::runtime_error - If inputs are invalid or a CUDA operation
   * fails.
   */
  void matvecT(const DeviceCsrMatrix& mat,
               DeviceConstVectorView  x,
               DeviceVectorView       y,
               Real                   alpha = 1.0,
               Real                   beta  = 0.0) const;

  /**
   * @brief Resize the output if needed and enqueue `out = mat * x`.
   *
   * @param[in] mat - Device CSR matrix.
   * @param[in] x - Device input vector.
   * @param[out] out - Resized Device output vector.
   * @throws std::runtime_error - If inputs are invalid or a CUDA operation
   * fails.
   */
  void matvec(const DeviceCsrMatrix& mat,
              DeviceConstVectorView  x,
              DeviceVector&          out) const;

  /**
   * @brief Resize the output if needed and enqueue `out = mat^T * x`.
   *
   * @param[in] mat - Device CSR matrix.
   * @param[in] x - Device input vector.
   * @param[out] out - Resized Device output vector.
   * @throws std::runtime_error - If inputs are invalid or a CUDA operation
   * fails.
   */
  void matvecT(const DeviceCsrMatrix& mat,
               DeviceConstVectorView  x,
               DeviceVector&          out) const;

  /**
   * @brief Enqueue `y = alpha * mat * x + beta * y` for a dense view.
   *
   * @param[in] mat - Device dense matrix view.
   * @param[in] x - Device input vector.
   * @param[in,out] y - Device output scaled by `beta` before accumulation.
   * @param[in] alpha - Matrix-product scale.
   * @param[in] beta - Existing-output scale.
   * @throws std::runtime_error - If inputs are invalid or a CUDA operation
   * fails.
   */
  void matvec(DeviceMatrixView<const Real> mat,
              DeviceConstVectorView        x,
              DeviceVectorView             y,
              Real                         alpha = 1.0,
              Real                         beta  = 0.0) const;

  /**
   * @brief Enqueue `y = alpha * mat^T * x + beta * y` for a dense view.
   *
   * @param[in] mat - Device dense matrix view.
   * @param[in] x - Device input vector.
   * @param[in,out] y - Device output scaled by `beta` before accumulation.
   * @param[in] alpha - Matrix-product scale.
   * @param[in] beta - Existing-output scale.
   * @throws std::runtime_error - If inputs are invalid or a CUDA operation
   * fails.
   */
  void matvecT(DeviceMatrixView<const Real> mat,
               DeviceConstVectorView        x,
               DeviceVectorView             y,
               Real                         alpha = 1.0,
               Real                         beta  = 0.0) const;

  /**
   * @brief Leave a Device CSR matrix unchanged after assembly.
   *
   * @param[in,out] mat - Assembled matrix.
   */
  void finalize(DeviceCsrMatrix& mat) const noexcept
  {
    static_cast<void>(mat);
  }

private:
  template <MemorySpace SrcSpace, MemorySpace DstSpace>
  static void checkCopy(const CsrMatrix<SrcSpace>& src,
                        const CsrMatrix<DstSpace>& dst)
  {
    require(src.rows() == dst.rows() && src.cols() == dst.cols()
                && src.nnz() == dst.nnz()
                && src.pattern().layoutId() == dst.pattern().layoutId()
                && src.vals().size() == src.nnz()
                && dst.vals().size() == dst.nnz(),
            "CsrMatrix copy requires compatible source and destination graphs");
  }

  CudaContext&      ctx_;         ///< Bound CUDA context.
  CudaVectorHandler vec_handler_; ///< CUDA vector operations.
};

using HostMatrixHandler = MatrixHandler<HostCsrBackend>;
using CudaMatrixHandler = MatrixHandler<CudaCsrBackend>;

} // namespace femx::linalg
