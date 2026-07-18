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
struct BoundaryMapView
{
  Index num_rows{0}; ///< CSR row count.
  Index num_cols{0}; ///< CSR column count.
  Index nnz{0};      ///< Number of CSR entries.
  Index num_bcs{0};  ///< Number of constrained DOFs.

  const Index* bc_rows{nullptr};      ///< Constrained rows in input order.
  const Index* diag_entries{nullptr}; ///< Diagonal CSR index per constraint.
  const Index* col_offsets{nullptr};  ///< Offsets into constrained columns.
  const Index* col_entries{nullptr};  ///< CSR entries in constrained columns.
  const Index* col_rows{nullptr};     ///< Row of each constrained entry.
  const Index* bc_mask{nullptr};      ///< Nonzero for constrained rows.

  /** @brief Return the constrained row at boundary index `ib`. */
  FEMX_HOST_DEVICE Index bcRow(Index ib) const
  {
    return bc_rows[ib];
  }

  /** @brief Return its diagonal position in the CSR value array. */
  FEMX_HOST_DEVICE Index diag(Index ib) const
  {
    return diag_entries[ib];
  }

  /** @brief Return the first constrained-column entry for boundary `ib`. */
  FEMX_HOST_DEVICE Index colBegin(Index ib) const
  {
    return col_offsets[ib];
  }

  /** @brief Return one-past-last constrained-column entry for boundary `ib`. */
  FEMX_HOST_DEVICE Index colEnd(Index ib) const
  {
    return col_offsets[ib + 1];
  }

  /** @brief Return whether a global row is constrained. */
  FEMX_HOST_DEVICE bool isBc(Index row) const
  {
    return bc_mask[row] != 0;
  }
};

/**
 * @brief Precomputed CSR locations used to enforce essential boundaries.
 *
 * The map preserves the constrained-DOF input order. Prescribed values passed
 * to prepareForwardSolve() use the same order. The map refers to one CSR
 * layout; matrices passed to the operations must use that layout.
 */
template <MemorySpace Space>
class BoundaryMap
{
public:
  /** @brief Index-vector type in this map's memory space. */
  using IndexVector = Vector<Space, Index>;

  BoundaryMap() = default;

  BoundaryMap(const BoundaryMap&)                = default;
  BoundaryMap(BoundaryMap&&) noexcept            = default;
  BoundaryMap& operator=(const BoundaryMap&)     = default;
  BoundaryMap& operator=(BoundaryMap&&) noexcept = default;

private:
  BoundaryMap(Index         num_rows,
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

  friend BoundaryMap<MemorySpace::Host> makeBoundaryMap(
      const Array<Index>& dofs,
      const HostCsrGraph& graph);

  friend void copy(const BoundaryMap<MemorySpace::Host>& src,
                   BoundaryMap<MemorySpace::Device>&     dst,
                   CudaContext&                          ctx);

public:
  /** @brief Return the mapped CSR row count. */
  Index rows() const noexcept
  {
    return num_rows_;
  }

  /** @brief Return the mapped CSR column count. */
  Index cols() const noexcept
  {
    return num_cols_;
  }

  /** @brief Return the mapped CSR entry count. */
  Index nnz() const noexcept
  {
    return nnz_;
  }

  /** @brief Return the identifier of the compatible CSR layout. */
  std::uint64_t layoutId() const noexcept
  {
    return layout_id_;
  }

  /** @brief Return the number of constrained DOFs. */
  Index numBcs() const noexcept
  {
    return bc_rows_.size();
  }

  /** @brief Return a non-owning kernel view valid while this map is alive. */
  BoundaryMapView<Space> view() const noexcept
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

using HostBoundaryMap   = BoundaryMap<MemorySpace::Host>;
using DeviceBoundaryMap = BoundaryMap<MemorySpace::Device>;

/**
 * @brief Build and validate boundary metadata for a host CSR graph.
 * @param dofs Unique constrained DOFs, whose order defines `bc_vals` order.
 * @param graph Square CSR graph used by all later boundary operations.
 */
HostBoundaryMap makeBoundaryMap(const Array<Index>& dofs,
                                const HostCsrGraph& graph);

/**
 * @brief Explicitly copy host boundary metadata to device storage.
 * @param src Validated host map.
 * @param dst Device map replaced by the copy.
 * @param ctx CUDA stream on which copies are enqueued.
 */
void copy(const HostBoundaryMap& src,
          DeviceBoundaryMap&     dst,
          CudaContext&           ctx);

/**
 * @brief Replace constrained rows in the authoritative Jacobian.
 *
 * Constrained rows are zeroed and their diagonal is set to `diag`. Columns
 * are deliberately left unchanged so the matrix continues to represent the
 * Jacobian of the row-replaced residual and can be used by adjoint solves.
 * Use one for a next-state Jacobian and zero for a history Jacobian.
 * @param map Boundary map matching `jac.graph()`.
 * @param jac Matrix modified in place.
 * @param diag Replacement diagonal value.
 */
void replaceRows(const HostBoundaryMap& map,
                 HostCsrMatrix&         jac,
                 Real                   diag);

/**
 * @brief Asynchronous CUDA equivalent of replaceRows().
 * @param map Device boundary map matching `jac.graph()`.
 * @param jac Device matrix modified in place.
 * @param diag Replacement diagonal value.
 * @param ctx CUDA stream on which work is enqueued.
 */
void replaceRows(const DeviceBoundaryMap& map,
                 DeviceCsrMatrix&         jac,
                 Real                     diag,
                 CudaContext&             ctx);

/**
 * @brief Replace constrained residual entries with state minus prescribed.
 * @param map Boundary map defining constrained rows and value order.
 * @param state Full state vector.
 * @param bc_vals Prescribed values in map order.
 * @param res Residual modified at constrained rows.
 */
void replaceRes(const HostBoundaryMap& map,
                const HostVector&      state,
                const HostVector&      bc_vals,
                HostVector&            res);

/**
 * @brief Asynchronous CUDA equivalent of replaceRes().
 * @param map Device boundary map.
 * @param state Full device state vector.
 * @param bc_vals Prescribed device values in map order.
 * @param res Device residual modified at constrained rows.
 * @param ctx CUDA stream on which work is enqueued.
 */
void replaceRes(const DeviceBoundaryMap& map,
                const DeviceVector&      state,
                const DeviceVector&      bc_vals,
                DeviceVector&            res,
                CudaContext&             ctx);

/**
 * @brief Prepare a separate forward-solve matrix and right-hand side.
 *
 * Constrained columns are eliminated with the corresponding RHS correction,
 * constrained rows are replaced by identity rows, and constrained RHS entries
 * are set to bc_vals. The input matrix must be a separate solve copy
 * when the authoritative row-replaced Jacobian is still needed.
 * @param map Boundary map matching `solve_mat.graph()`.
 * @param solve_mat Matrix modified into the forward system.
 * @param rhs Right-hand side corrected and constrained in place.
 * @param bc_vals Prescribed values in map order.
 */
void prepareForwardSolve(const HostBoundaryMap& map,
                         HostCsrMatrix&         solve_mat,
                         HostVector&            rhs,
                         const HostVector&      bc_vals);

/**
 * @brief Asynchronous CUDA equivalent of prepareForwardSolve().
 * @param map Device boundary map matching `solve_mat.graph()`.
 * @param solve_mat Device matrix modified into the forward system.
 * @param rhs Device right-hand side modified in place.
 * @param bc_vals Prescribed device values in map order.
 * @param ctx CUDA stream on which work is enqueued.
 */
void prepareForwardSolve(const DeviceBoundaryMap& map,
                         DeviceCsrMatrix&         solve_mat,
                         DeviceVector&            rhs,
                         const DeviceVector&      bc_vals,
                         CudaContext&             ctx);

} // namespace assembly
} // namespace femx
