#pragma once

#include <cstdint>

#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/SystemMatrix.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>

namespace femx::linalg
{

/**
 * @brief Own and operate on a Device CSR system matrix.
 *
 * Assembly, constraint kernels, matrix application, and copies use the stream
 * owned by the bound CUDA context. Constraint metadata is cached in Device
 * storage.
 */
class CudaSystemMatrix final : public SystemMatrix<MemorySpace::Device>
{
public:
  /**
   * @brief Bind the system matrix to a CUDA execution context.
   *
   * @param[in] ctx - Context providing the stream and CUDA handles.
   */
  explicit CudaSystemMatrix(CudaContext& ctx) noexcept;

  /**
   * @brief Prepare zero-valued Device storage from a Host CSR pattern.
   *
   * @param[in] pattern - Canonical global sparsity pattern.
   */
  void setup(const HostCsrPattern& pattern) override;

  /**
   * @brief Replace constrained rows by diagonal rows.
   *
   * @param[in] rows - Device constrained row indices.
   * @param[in] diag - Replacement diagonal value.
   * @throws - If the constrained rows are invalid.
   */
  void replaceRows(DeviceVectorView<const Index> rows,
                   Real                          diag) override;

  /**
   * @brief Eliminate constrained columns and correct a right-hand side.
   *
   * @param[in]     rows - Device constrained row indices.
   * @param[in]     vals - Device prescribed values.
   * @param[in,out] rhs - Device right-hand side corrected in place.
   * @throws - If the constraint vectors are incompatible.
   */
  void eliminateColumns(DeviceVectorView<const Index> rows,
                        DeviceVectorView<const Real>  vals,
                        DeviceVectorView<Real>        rhs) override;

  /**
   * @brief Complete assembly before matrix application.
   */
  void finalize() override;

  /**
   * @brief Compute the Device system-matrix product.
   *
   * @param[in]  dir - Device input direction.
   * @param[out] out - Resized Device output vector.
   */
  void apply(DeviceVectorView<const Real> dir,
             DeviceVector<Real>&          out) const override;

  /**
   * @brief Compute the transposed Device system-matrix product.
   *
   * @param[in]  dir - Device input direction.
   * @param[out] out - Resized Device output vector.
   */
  void applyT(DeviceVectorView<const Real> dir,
              DeviceVector<Real>&          out) const override;

  /**
   * @brief Return Device CSR storage for an assembly kernel.
   */
  DeviceCsrAssemblyView assemblyView() noexcept;

  /**
   * @brief Return the owned CSR matrix for a native Device solver.
   */
  const DeviceCsrMatrix& matrix() const noexcept;

  /**
   * @brief Construct or update a Device CSR transpose.
   *
   * @param[in]     src - Source matrix.
   * @param[in,out] dst - Transposed destination.
   * @throws - If `src` and `dst` are the same matrix.
   */
  void transpose(const DeviceCsrMatrix& src,
                 DeviceCsrMatrix&       dst) const;

  /**
   * @brief Apply an arbitrary Device CSR matrix.
   *
   * Compute `out = alpha * mat * dir + beta * out`.
   *
   * @param[in]     mat - Device CSR matrix.
   * @param[in]     dir - Input direction.
   * @param[in,out] out - Output vector.
   * @param[in]     alpha - Matrix-product scale.
   * @param[in]     beta - Existing-output scale.
   * @throws - If dimensions or storage are incompatible.
   */
  void apply(const DeviceCsrMatrix&       mat,
             DeviceVectorView<const Real> dir,
             DeviceVectorView<Real>       out,
             Real                         alpha = 1.0,
             Real                         beta  = 0.0) const;

  /**
   * @brief Apply the transpose of an arbitrary Device CSR matrix.
   *
   * Compute `out = alpha * transpose(mat) * dir + beta * out`.
   *
   * @param[in]     mat - Device CSR matrix.
   * @param[in]     dir - Input direction.
   * @param[in,out] out - Output vector.
   * @param[in]     alpha - Matrix-product scale.
   * @param[in]     beta - Existing-output scale.
   * @throws - If dimensions or storage are incompatible.
   */
  void applyT(const DeviceCsrMatrix&       mat,
              DeviceVectorView<const Real> dir,
              DeviceVectorView<Real>       out,
              Real                         alpha = 1.0,
              Real                         beta  = 0.0) const;

  /**
   * @brief Apply a row-major dense Device matrix.
   *
   * Compute `out = alpha * mat * dir + beta * out`.
   *
   * @param[in]     mat - Row-major dense Device matrix.
   * @param[in]     dir - Input direction.
   * @param[in,out] out - Output vector.
   * @param[in]     alpha - Matrix-product scale.
   * @param[in]     beta - Existing-output scale.
   * @throws - If dimensions or storage are incompatible.
   */
  void apply(DeviceMatrixView<const Real> mat,
             DeviceVectorView<const Real> dir,
             DeviceVectorView<Real>       out,
             Real                         alpha = 1.0,
             Real                         beta  = 0.0) const;

  /**
   * @brief Apply the transpose of a row-major dense Device matrix.
   *
   * Compute `out = alpha * transpose(mat) * dir + beta * out`.
   *
   * @param[in]     mat - Row-major dense Device matrix.
   * @param[in]     dir - Input direction.
   * @param[in,out] out - Output vector.
   * @param[in]     alpha - Matrix-product scale.
   * @param[in]     beta - Existing-output scale.
   * @throws - If dimensions or storage are incompatible.
   */
  void applyT(DeviceMatrixView<const Real> mat,
              DeviceVectorView<const Real> dir,
              DeviceVectorView<Real>       out,
              Real                         alpha = 1.0,
              Real                         beta  = 0.0) const;

private:
  struct ConstraintCache
  {
    std::uint64_t       layout_id{0};      ///< Cached CSR layout identifier.
    const Index*        rows{nullptr};     ///< Cached constrained-row address.
    Index               count{0};          ///< Number of constrained rows.
    DeviceVector<Index> row_to_constraint; ///< Row-to-constraint mapping.
  };

  void ensureConstraints(DeviceVectorView<const Index> rows);

  CudaContext&    ctx_;         ///< Bound CUDA execution context.
  DeviceCsrMatrix mat_;         ///< Owned Device CSR matrix.
  ConstraintCache constraints_; ///< Cached constraint metadata.
};

} // namespace femx::linalg
