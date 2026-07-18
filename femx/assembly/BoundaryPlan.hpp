#pragma once

#include <cstdint>
#include <utility>

#include <femx/common/Context.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/CsrGraph.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace assembly
{

/** @brief Non-owning boundary metadata consumed by host and CUDA kernels. */
template <MemorySpace Space>
struct BoundaryPlanView
{
  Index num_rows{0};
  Index num_cols{0};
  Index nnz{0};
  Index num_bcs{0};

  const Index* bc_rows{nullptr};
  const Index* diag_entries{nullptr};
  const Index* col_offsets{nullptr};
  const Index* col_entries{nullptr};
  const Index* col_rows{nullptr};
  const Index* bc_mask{nullptr};

  FEMX_HOST_DEVICE Index bcRow(Index ib) const
  {
    return bc_rows[ib];
  }

  FEMX_HOST_DEVICE Index diag(Index ib) const
  {
    return diag_entries[ib];
  }

  FEMX_HOST_DEVICE Index colBegin(Index ib) const
  {
    return col_offsets[ib];
  }

  FEMX_HOST_DEVICE Index colEnd(Index ib) const
  {
    return col_offsets[ib + 1];
  }

  FEMX_HOST_DEVICE bool isBc(Index row) const
  {
    return bc_mask[row] != 0;
  }
};

/**
 * @brief Precomputed CSR locations used to enforce essential boundaries.
 *
 * The plan preserves the constrained-DOF input order. Prescribed values passed
 * to prepareForwardSolve() use the same order. The plan refers to one CSR
 * layout; matrices passed to the operations must use that layout.
 */
template <MemorySpace Space>
class BoundaryPlan
{
public:
  using IndexVector = Vector<Space, Index>;

  BoundaryPlan() = default;

  BoundaryPlan(const BoundaryPlan&)                = default;
  BoundaryPlan(BoundaryPlan&&) noexcept            = default;
  BoundaryPlan& operator=(const BoundaryPlan&)     = default;
  BoundaryPlan& operator=(BoundaryPlan&&) noexcept = default;

private:
  BoundaryPlan(Index         num_rows,
               Index         num_cols,
               Index         nnz,
               std::uint64_t layout_id,
               IndexVector   bc_rows,
               IndexVector   diag,
               IndexVector   col_offsets,
               IndexVector   col_entries,
               IndexVector   col_rows,
               IndexVector   bc_mask)
    : num_rows_(num_rows),
      num_cols_(num_cols),
      nnz_(nnz),
      layout_id_(layout_id),
      bc_rows_(std::move(bc_rows)),
      diag_(std::move(diag)),
      col_offsets_(std::move(col_offsets)),
      col_entries_(std::move(col_entries)),
      col_rows_(std::move(col_rows)),
      bc_mask_(std::move(bc_mask))
  {
  }

  friend BoundaryPlan<MemorySpace::Host> makeBoundaryPlan(
      const Array<Index>&,
      const HostCsrGraph&);

  friend void copy(const BoundaryPlan<MemorySpace::Host>&,
                   BoundaryPlan<MemorySpace::Device>&,
                   CudaContext&);

public:
  Index rows() const noexcept
  {
    return num_rows_;
  }

  Index cols() const noexcept
  {
    return num_cols_;
  }

  Index nnz() const noexcept
  {
    return nnz_;
  }

  std::uint64_t layoutId() const noexcept
  {
    return layout_id_;
  }

  Index numBcs() const noexcept
  {
    return bc_rows_.size();
  }

  BoundaryPlanView<Space> view() const noexcept
  {
    return {num_rows_,
            num_cols_,
            nnz_,
            numBcs(),
            bc_rows_.data(),
            diag_.data(),
            col_offsets_.data(),
            col_entries_.data(),
            col_rows_.data(),
            bc_mask_.data()};
  }

private:
  Index         num_rows_{0};
  Index         num_cols_{0};
  Index         nnz_{0};
  std::uint64_t layout_id_{0};

  IndexVector bc_rows_;
  IndexVector diag_;
  IndexVector col_offsets_;
  IndexVector col_entries_;
  IndexVector col_rows_;
  IndexVector bc_mask_;
};

using HostBoundaryPlan   = BoundaryPlan<MemorySpace::Host>;
using DeviceBoundaryPlan = BoundaryPlan<MemorySpace::Device>;

/** @brief Build and validate boundary metadata for a host CSR graph. */
HostBoundaryPlan makeBoundaryPlan(const Array<Index>& dofs,
                                  const HostCsrGraph& graph);

/** @brief Explicitly copy host boundary metadata to device storage. */
void copy(const HostBoundaryPlan& src,
          DeviceBoundaryPlan&     dst,
          CudaContext&            ctx);

/**
 * @brief Replace constrained rows in the authoritative Jacobian.
 *
 * Constrained rows are zeroed and their diagonal is set to one. Columns are
 * deliberately left unchanged so the matrix continues to represent the
 * Jacobian of the row-replaced residual and can be used by adjoint solves.
 */
void replaceRows(const HostBoundaryPlan& plan, HostCsrMatrix& jac);

/** @brief Asynchronous CUDA equivalent of replaceRows(). */
void replaceRows(const DeviceBoundaryPlan& plan,
                 DeviceCsrMatrix&          jac,
                 CudaContext&              ctx);

/** @brief Replace constrained residual entries with state - prescribed. */
void replaceRes(const HostBoundaryPlan& plan,
                const HostVector&       state,
                const HostVector&       bc_vals,
                HostVector&             res);

/** @brief Asynchronous CUDA equivalent of replaceRes(). */
void replaceRes(const DeviceBoundaryPlan& plan,
                const DeviceVector&       state,
                const DeviceVector&       bc_vals,
                DeviceVector&             res,
                CudaContext&              ctx);

/**
 * @brief Prepare a separate forward-solve matrix and right-hand side.
 *
 * Constrained columns are eliminated with the corresponding RHS correction,
 * constrained rows are replaced by identity rows, and constrained RHS entries
 * are set to bc_vals. The input matrix must be a separate solve copy
 * when the authoritative row-replaced Jacobian is still needed.
 */
void prepareForwardSolve(const HostBoundaryPlan& plan,
                         HostCsrMatrix&          solve_mat,
                         HostVector&             rhs,
                         const HostVector&       bc_vals);

/** @brief Asynchronous CUDA equivalent of prepareForwardSolve(). */
void prepareForwardSolve(const DeviceBoundaryPlan& plan,
                         DeviceCsrMatrix&          solve_mat,
                         DeviceVector&             rhs,
                         const DeviceVector&       bc_vals,
                         CudaContext&              ctx);

} // namespace assembly
} // namespace femx
