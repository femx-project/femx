#include <stdexcept>
#include <string>
#include <utility>

#include <femx/assembly/BoundaryPlan.hpp>

namespace femx
{
namespace assembly
{
namespace
{

template <MemorySpace Space>
void checkMat(const BoundaryPlan<Space>& plan, const CsrMatrix<Space>& mat)
{
  if (mat.graph().layoutId() != plan.layoutId())
  {
    throw std::runtime_error(
        "BoundaryPlan matrix does not match the planned CSR layout");
  }
}

void checkForward(const HostBoundaryPlan& plan,
                  const HostCsrMatrix&    mat,
                  const HostVector&       rhs,
                  const HostVector&       bc_vals)
{
  checkMat(plan, mat);
  if (rhs.size() != plan.rows() || bc_vals.size() != plan.numBcs())
  {
    throw std::runtime_error(
        "BoundaryPlan forward vectors have incompatible sizes");
  }
  if (&rhs == &bc_vals)
  {
    throw std::runtime_error(
        "BoundaryPlan RHS and prescribed values must not alias");
  }
  const HostVector& mat_vals = mat.vals();
  if (&rhs == &mat_vals || &bc_vals == &mat_vals)
  {
    throw std::runtime_error(
        "BoundaryPlan vectors must not alias matrix values");
  }
}

void replaceRowsRaw(const HostBoundaryPlan& plan, HostCsrMatrix& mat)
{
  const auto  view    = plan.view();
  const auto& row_ptr = mat.graph().rowPtr();
  Real*       vals    = mat.valsData();

  for (Index ib = 0; ib < view.num_bcs; ++ib)
  {
    const Index row = view.bcRow(ib);
    for (Index k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
    {
      vals[k] = 0.0;
    }
    vals[view.diag(ib)] = 1.0;
  }
}

} // namespace

HostBoundaryPlan makeBoundaryPlan(const Array<Index>& dofs,
                                  const HostCsrGraph& graph)
{
  if (graph.rows() != graph.cols())
  {
    throw std::runtime_error("BoundaryPlan requires a square CSR graph");
  }

  const Index     num_bcs = dofs.size();
  HostIndexVector bc_rows(num_bcs);
  HostIndexVector diag(num_bcs, -1);
  HostIndexVector col_offsets(num_bcs + 1, 0);
  HostIndexVector bc_mask(graph.rows(), 0);
  Array<Index>    bc_by_col(graph.cols(), -1);

  for (Index ib = 0; ib < num_bcs; ++ib)
  {
    const Index dof = dofs[ib];
    if (dof < 0 || dof >= graph.rows())
    {
      throw std::runtime_error(
          "BoundaryPlan constrained DOF is out of range");
    }
    if (bc_mask[dof] != 0)
    {
      throw std::runtime_error(
          "BoundaryPlan constrained DOFs must be unique");
    }
    bc_rows[ib]    = dof;
    bc_mask[dof]   = 1;
    bc_by_col[dof] = ib;
  }

  const auto& row_ptr = graph.rowPtr();
  const auto& cols    = graph.colInd();
  for (Index row = 0; row < graph.rows(); ++row)
  {
    for (Index k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
    {
      const Index col = cols[k];
      const Index ib  = bc_by_col[col];
      if (ib >= 0)
      {
        ++col_offsets[ib + 1];
        if (row == col)
        {
          if (diag[ib] >= 0)
          {
            throw std::runtime_error(
                "BoundaryPlan constrained row has duplicate diagonal entries");
          }
          diag[ib] = k;
        }
      }
    }
  }

  for (Index ib = 0; ib < num_bcs; ++ib)
  {
    if (diag[ib] < 0)
    {
      throw std::runtime_error(
          "BoundaryPlan constrained row has no diagonal entry");
    }
    col_offsets[ib + 1] += col_offsets[ib];
  }

  HostIndexVector col_entries(col_offsets[num_bcs]);
  HostIndexVector col_rows(col_offsets[num_bcs]);
  HostIndexVector next = col_offsets;
  for (Index row = 0; row < graph.rows(); ++row)
  {
    for (Index k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
    {
      const Index ib = bc_by_col[cols[k]];
      if (ib >= 0)
      {
        const Index dst  = next[ib]++;
        col_entries[dst] = k;
        col_rows[dst]    = row;
      }
    }
  }

  return {graph.rows(),
          graph.cols(),
          graph.nnz(),
          graph.layoutId(),
          std::move(bc_rows),
          std::move(diag),
          std::move(col_offsets),
          std::move(col_entries),
          std::move(col_rows),
          std::move(bc_mask)};
}

void copy(const HostBoundaryPlan& src,
          DeviceBoundaryPlan&     dst,
          CudaContext&            ctx)
{
  DeviceIndexVector bc_rows;
  DeviceIndexVector diag;
  DeviceIndexVector col_offsets;
  DeviceIndexVector col_entries;
  DeviceIndexVector col_rows;
  DeviceIndexVector bc_mask;

  femx::copy(src.bc_rows_, bc_rows, ctx);
  femx::copy(src.diag_, diag, ctx);
  femx::copy(src.col_offsets_, col_offsets, ctx);
  femx::copy(src.col_entries_, col_entries, ctx);
  femx::copy(src.col_rows_, col_rows, ctx);
  femx::copy(src.bc_mask_, bc_mask, ctx);

  dst = {src.num_rows_,
         src.num_cols_,
         src.nnz_,
         src.layout_id_,
         std::move(bc_rows),
         std::move(diag),
         std::move(col_offsets),
         std::move(col_entries),
         std::move(col_rows),
         std::move(bc_mask)};
}

void replaceRows(const HostBoundaryPlan& plan, HostCsrMatrix& jac)
{
  checkMat(plan, jac);
  replaceRowsRaw(plan, jac);
}

void replaceRes(const HostBoundaryPlan& plan,
                const HostVector&       state,
                const HostVector&       bc_vals,
                HostVector&             res)
{
  if (state.size() != plan.rows() || res.size() != plan.rows()
      || bc_vals.size() != plan.numBcs())
  {
    throw std::runtime_error(
        "BoundaryPlan residual vectors have incompatible sizes");
  }
  if (&state == &res || &bc_vals == &res)
  {
    throw std::runtime_error(
        "BoundaryPlan residual output must not alias its inputs");
  }
  const auto view = plan.view();
  for (Index ib = 0; ib < view.num_bcs; ++ib)
  {
    const Index row = view.bcRow(ib);
    res[row]        = state[row] - bc_vals[ib];
  }
}

void prepareForwardSolve(const HostBoundaryPlan& plan,
                         HostCsrMatrix&          solve_mat,
                         HostVector&             rhs,
                         const HostVector&       bc_vals)
{
  checkForward(plan, solve_mat, rhs, bc_vals);

  const auto view = plan.view();
  Real*      vals = solve_mat.valsData();
  for (Index ib = 0; ib < view.num_bcs; ++ib)
  {
    const Real bc = bc_vals[ib];
    for (Index i = view.colBegin(ib); i < view.colEnd(ib); ++i)
    {
      const Index row = view.col_rows[i];
      const Index k   = view.col_entries[i];
      if (!view.isBc(row))
      {
        rhs[row] -= vals[k] * bc;
      }
      vals[k] = 0.0;
    }
  }

  replaceRowsRaw(plan, solve_mat);
  for (Index ib = 0; ib < view.num_bcs; ++ib)
  {
    rhs[view.bcRow(ib)] = bc_vals[ib];
  }
}

#if !defined(FEMX_HAS_CUDA)
void replaceRows(const DeviceBoundaryPlan&,
                 DeviceCsrMatrix&,
                 CudaContext&)
{
  throw std::runtime_error(
      "BoundaryPlan CUDA operations require FEMX_ENABLE_CUDA");
}

void replaceRes(const DeviceBoundaryPlan&,
                const DeviceVector&,
                const DeviceVector&,
                DeviceVector&,
                CudaContext&)
{
  throw std::runtime_error(
      "BoundaryPlan CUDA operations require FEMX_ENABLE_CUDA");
}

void prepareForwardSolve(const DeviceBoundaryPlan&,
                         DeviceCsrMatrix&,
                         DeviceVector&,
                         const DeviceVector&,
                         CudaContext&)
{
  throw std::runtime_error(
      "BoundaryPlan CUDA operations require FEMX_ENABLE_CUDA");
}
#endif

} // namespace assembly
} // namespace femx
