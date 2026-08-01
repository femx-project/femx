#pragma once

#include <cstdint>

#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/SystemMatrix.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>

namespace femx::linalg
{

/**
 * @brief Own and operate on a Device CSR system matrix.
 */
class CudaSystemMatrix final : public SystemMatrix<MemorySpace::Device>
{
  using Base = SystemMatrix<MemorySpace::Device>;

public:
  /**
   * @brief Bind the system matrix to a CUDA execution context.
   *
   * @param[in] ctx - Context providing the stream and CUDA handles.
   */
  explicit CudaSystemMatrix(CudaContext& ctx) noexcept;

  /**
   * @copydoc Base::setup()
   */
  void setup(const HostCsrPattern& pattern) override;

  /**
   * @copydoc Base::replaceRows()
   *
   * @throws std::runtime_error If validation fails.
   */
  void replaceRows(DeviceVectorView<const Index> rows,
                   Real                          diag) override;

  /**
   * @copydoc Base::eliminateColumns()
   *
   * @throws std::runtime_error If validation fails.
   */
  void eliminateColumns(DeviceVectorView<const Index> rows,
                        DeviceVectorView<const Real>  values,
                        DeviceVectorView<Real>        rhs) override;

  /**
   * @copydoc Base::finalize()
   */
  void finalize() override;

  /**
   * @copydoc Base::matvec()
   */
  void matvec(DeviceVectorView<const Real> dir,
              DeviceVector<Real>&          out) const override;

  /**
   * @copydoc Base::matvecT()
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
