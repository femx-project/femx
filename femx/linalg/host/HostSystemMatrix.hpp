#pragma once

#include <memory>

#include <femx/linalg/Context.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/SystemMatrix.hpp>

namespace femx::linalg
{

/**
 * @brief Own and operate on a Host CSR system matrix.
 *
 * Constraint metadata derived from the current pattern and constrained rows
 * is cached by this object.
 */
class HostSystemMatrix final : public SystemMatrix<MemorySpace::Host>
{
public:
  /**
   * @brief Bind the system matrix to a Host execution context.
   *
   * @param[in] ctx - Context providing Host vector operations.
   */
  explicit HostSystemMatrix(Context<MemorySpace::Host>& ctx) noexcept;

  ~HostSystemMatrix() override;

  HostSystemMatrix(const HostSystemMatrix&)            = delete;
  HostSystemMatrix& operator=(const HostSystemMatrix&) = delete;
  HostSystemMatrix(HostSystemMatrix&&)                 = delete;
  HostSystemMatrix& operator=(HostSystemMatrix&&)      = delete;

  /**
   * @brief Prepare zero-valued storage for a Host CSR pattern.
   *
   * @param[in] pattern - Canonical global sparsity pattern.
   */
  void setup(const HostCsrPattern& pattern) override;

  /**
   * @brief Add one element contribution.
   *
   * @param[in] elem - Element rows, columns, CSR entries, and values.
   * @throws - If the element views are incompatible.
   */
  void addElement(const ElementJacobianView& elem) override;

  /**
   * @brief Replace constrained rows by diagonal rows.
   *
   * @param[in] rows - Constrained global row indices.
   * @param[in] diag - Replacement diagonal value.
   * @throws - If the constrained rows are invalid.
   */
  void replaceRows(HostVectorView<const Index> rows,
                   Real                        diag) override;

  /**
   * @brief Eliminate constrained columns and correct a right-hand side.
   *
   * @param[in]     rows - Constrained global row indices.
   * @param[in]     vals - Prescribed values.
   * @param[in,out] rhs - Right-hand side corrected in place.
   * @throws - If the constraint vectors are incompatible.
   */
  void eliminateColumns(HostVectorView<const Index> rows,
                        HostVectorView<const Real>  vals,
                        HostVectorView<Real>        rhs) override;

  /**
   * @brief Complete assembly before matrix application.
   */
  void finalize() override;

  /**
   * @brief Compute the Host system-matrix product.
   *
   * @param[in]  dir - Input direction.
   * @param[out] out - Resized output vector.
   */
  void matvec(HostVectorView<const Real> dir,
              HostVector<Real>&          out) const override;

  /**
   * @brief Compute the transposed Host system-matrix product.
   *
   * @param[in]  dir - Input direction.
   * @param[out] out - Resized output vector.
   */
  void matvecT(HostVectorView<const Real> dir,
               HostVector<Real>&          out) const override;

  /**
   * @brief Return the owned CSR matrix for a native Host solver.
   */
  const HostCsrMatrix& matrix() const noexcept;

private:
  class ConstraintCache;

  ConstraintCache& constraints(HostVectorView<const Index> rows);

  Context<MemorySpace::Host>&      ctx_;         ///< Bound Host execution context.
  HostCsrMatrix                    mat_;         ///< Owned Host CSR matrix.
  std::unique_ptr<ConstraintCache> constraints_; ///< Cached constraint metadata.
};

} // namespace femx::linalg
