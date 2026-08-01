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
  void matvec(DeviceVectorView<const Real> dir,
              DeviceVector<Real>&          out) const override;

  /**
   * @brief Compute the transposed Device system-matrix product.
   *
   * @param[in]  dir - Device input direction.
   * @param[out] out - Resized Device output vector.
   */
  void matvecT(DeviceVectorView<const Real> dir,
               DeviceVector<Real>&          out) const override;

  /**
   * @brief Return Device CSR storage for an assembly kernel.
   */
  DeviceCsrAssemblyView assemblyView() noexcept;

  /**
   * @brief Return the owned CSR matrix for a native Device solver.
   */
  const DeviceCsrMatrix& matrix() const noexcept;

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
