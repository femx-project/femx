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

  void setup(const HostCsrPattern& pattern) override;
  void replaceRows(DeviceVectorView<const Index> rows,
                   Real                          diagonal) override;
  void eliminateColumns(DeviceVectorView<const Index> rows,
                        DeviceVectorView<const Real>  values,
                        DeviceVectorView<Real>        rhs) override;
  void finalize() override;
  void apply(DeviceVectorView<const Real> direction,
             DeviceVector<Real>&          out) const override;
  void applyT(DeviceVectorView<const Real> direction,
              DeviceVector<Real>&          out) const override;

  /** @brief Return Device CSR storage for an assembly kernel. */
  DeviceCsrAssemblyView assemblyView() noexcept;

  /** @brief Return the owned CSR matrix for a native Device solver. */
  const DeviceCsrMatrix& matrix() const noexcept;

  /**
   * @brief Construct or update a Device CSR transpose.
   *
   * @param[in] source - Source matrix.
   * @param[in,out] destination - Transposed destination.
   */
  void transpose(const DeviceCsrMatrix& source,
                 DeviceCsrMatrix&       destination) const;

  /** @brief Apply an arbitrary Device CSR matrix. */
  void apply(const DeviceCsrMatrix&       matrix,
             DeviceVectorView<const Real> direction,
             DeviceVectorView<Real>       out,
             Real                         alpha = 1.0,
             Real                         beta  = 0.0) const;

  /** @brief Apply the transpose of an arbitrary Device CSR matrix. */
  void applyT(const DeviceCsrMatrix&       matrix,
              DeviceVectorView<const Real> direction,
              DeviceVectorView<Real>       out,
              Real                         alpha = 1.0,
              Real                         beta  = 0.0) const;

  /** @brief Apply a row-major dense Device matrix. */
  void apply(DeviceMatrixView<const Real> matrix,
             DeviceVectorView<const Real> direction,
             DeviceVectorView<Real>       out,
             Real                         alpha = 1.0,
             Real                         beta  = 0.0) const;

  /** @brief Apply the transpose of a row-major dense Device matrix. */
  void applyT(DeviceMatrixView<const Real> matrix,
              DeviceVectorView<const Real> direction,
              DeviceVectorView<Real>       out,
              Real                         alpha = 1.0,
              Real                         beta  = 0.0) const;

private:
  struct ConstraintCache
  {
    std::uint64_t       layout_id{0};
    const Index*        rows{nullptr};
    Index               count{0};
    DeviceVector<Index> row_to_constraint;
  };

  void ensureConstraints(DeviceVectorView<const Index> rows);

  CudaContext&    ctx_;
  DeviceCsrMatrix matrix_;
  ConstraintCache constraints_;
};

} // namespace femx::linalg
