#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/fem/DofLayout.hpp>

namespace femx
{
namespace assembly
{
namespace
{

Index checkedMul(Index lhs, Index rhs)
{
  if (lhs < 0 || rhs < 0)
  {
    throw std::runtime_error(
        "AssemblyMap local dimensions must be non-negative");
  }
  const std::int64_t val = static_cast<std::int64_t>(lhs) * rhs;
  if (val > std::numeric_limits<Index>::max())
  {
    throw std::runtime_error("AssemblyMap local Jacobian is too large");
  }
  return static_cast<Index>(val);
}

Index checkedAdd(Index lhs, Index rhs)
{
  const std::int64_t val = static_cast<std::int64_t>(lhs) + rhs;
  if (val < 0 || val > std::numeric_limits<Index>::max())
  {
    throw std::runtime_error(
        "AssemblyMap flattened data exceeds the Index range");
  }
  return static_cast<Index>(val);
}

void checkDofs(const Array<Index>& dofs, Index size, const char* kind)
{
  for (Index dof : dofs)
  {
    if (dof < 0 || dof >= size)
    {
      throw std::runtime_error(std::string("AssemblyMap ") + kind
                               + " DOF is out of range");
    }
  }
}

} // namespace

AssemblyMap<MemorySpace::Host> makeAssemblyMap(
    Index                      num_res,
    Index                      num_states,
    const Array<Array<Index>>& elem_res,
    const Array<Array<Index>>& elem_state)
{
  if (num_res < 0 || num_states < 0)
  {
    throw std::runtime_error(
        "AssemblyMap global dimensions must be non-negative");
  }
  if (elem_res.size() != elem_state.size())
  {
    throw std::runtime_error(
        "AssemblyMap residual/state element counts differ");
  }

  const Index num_elem = elem_res.size();

  HostIndexVector res_offsets(num_elem + 1);
  HostIndexVector state_offsets(num_elem + 1);
  HostIndexVector jac_offsets(num_elem + 1);
  HostIndexVector res_dofs;
  HostIndexVector state_dofs;

  Index max_res_dofs    = 0;
  Index max_state_dofs  = 0;
  Index max_jac_entries = 0;

  for (Index ie = 0; ie < num_elem; ++ie)
  {
    const auto& rows = elem_res[ie];
    const auto& cols = elem_state[ie];
    checkDofs(rows, num_res, "residual");
    checkDofs(cols, num_states, "state");

    for (Index row : rows)
    {
      res_dofs.push_back(row);
    }
    for (Index col : cols)
    {
      state_dofs.push_back(col);
    }

    const Index elem_nnz  = checkedMul(rows.size(), cols.size());
    res_offsets[ie + 1]   = res_dofs.size();
    state_offsets[ie + 1] = state_dofs.size();
    jac_offsets[ie + 1]   = checkedAdd(jac_offsets[ie], elem_nnz);
    max_res_dofs          = std::max(max_res_dofs, rows.size());
    max_state_dofs        = std::max(max_state_dofs, cols.size());
    max_jac_entries       = std::max(max_jac_entries, elem_nnz);
  }

  const Index     nnz = jac_offsets[num_elem];
  HostIndexVector coo_rows(nnz);
  HostIndexVector coo_cols(nnz);
  HostIndexVector order(nnz);
  HostIndexVector jac_map(nnz);

  Index k = 0;
  for (Index ie = 0; ie < num_elem; ++ie)
  {
    for (Index row : elem_res[ie])
    {
      for (Index col : elem_state[ie])
      {
        coo_rows[k] = row;
        coo_cols[k] = col;
        order[k]    = k;
        ++k;
      }
    }
  }

  std::sort(order.begin(),
            order.end(),
            [&coo_rows, &coo_cols](Index lhs, Index rhs)
            {
              if (coo_rows[lhs] != coo_rows[rhs])
              {
                return coo_rows[lhs] < coo_rows[rhs];
              }
              return coo_cols[lhs] < coo_cols[rhs];
            });

  HostIndexVector row_ptr(num_res + 1, 0);
  HostIndexVector cols;
  cols.reserve(nnz);

  Index csr_i = -1;
  for (Index i = 0; i < nnz; ++i)
  {
    const Index curr   = order[i];
    const bool  is_new = i == 0 || coo_rows[curr] != coo_rows[order[i - 1]]
                        || coo_cols[curr] != coo_cols[order[i - 1]];
    if (is_new)
    {
      ++csr_i;
      cols.push_back(coo_cols[curr]);
      ++row_ptr[coo_rows[curr] + 1];
    }
    jac_map[curr] = csr_i;
  }

  for (Index row = 0; row < num_res; ++row)
  {
    row_ptr[row + 1] += row_ptr[row];
  }

  HostCsrGraph graph(num_res,
                     num_states,
                     std::move(row_ptr),
                     std::move(cols));

  return {num_elem,
          num_res,
          num_states,
          std::move(res_offsets),
          std::move(res_dofs),
          std::move(state_offsets),
          std::move(state_dofs),
          std::move(jac_offsets),
          std::move(jac_map),
          std::move(graph),
          max_res_dofs,
          max_state_dofs,
          max_jac_entries};
}

AssemblyMap<MemorySpace::Host> makeAssemblyMap(
    fem::DofLayout res_lyt,
    fem::DofLayout state_lyt)
{
  if (res_lyt.numElems() != state_lyt.numElems())
  {
    throw std::runtime_error(
        "AssemblyMap residual/state layouts have different element counts");
  }

  Array<Array<Index>> res_dofs(res_lyt.numElems());
  Array<Array<Index>> state_dofs(state_lyt.numElems());
  for (Index ie = 0; ie < res_lyt.numElems(); ++ie)
  {
    res_lyt.elemDofs(ie, res_dofs[ie]);
    state_lyt.elemDofs(ie, state_dofs[ie]);
  }
  return makeAssemblyMap(
      res_lyt.numDofs(), state_lyt.numDofs(), res_dofs, state_dofs);
}

AssemblyMap<MemorySpace::Host> makeAssemblyMap(fem::DofLayout layout)
{
  return makeAssemblyMap(layout, layout);
}

void copy(const AssemblyMap<MemorySpace::Host>& src,
          AssemblyMap<MemorySpace::Device>&     dst,
          CudaContext&                          ctx)
{
  DeviceIndexVector res_offsets;
  DeviceIndexVector res_dofs;
  DeviceIndexVector state_offsets;
  DeviceIndexVector state_dofs;
  DeviceIndexVector jac_offsets;
  DeviceIndexVector jac_map;
  DeviceCsrGraph    graph;

  femx::copy(src.res_offsets_, res_offsets, ctx);
  femx::copy(src.res_dofs_, res_dofs, ctx);
  femx::copy(src.state_offsets_, state_offsets, ctx);
  femx::copy(src.state_dofs_, state_dofs, ctx);
  femx::copy(src.jac_offsets_, jac_offsets, ctx);
  femx::copy(src.jac_map_, jac_map, ctx);
  femx::copy(src.graph_, graph, ctx);

  dst = {src.num_elems_,
         src.num_res_,
         src.num_states_,
         std::move(res_offsets),
         std::move(res_dofs),
         std::move(state_offsets),
         std::move(state_dofs),
         std::move(jac_offsets),
         std::move(jac_map),
         std::move(graph),
         src.max_res_dofs_,
         src.max_state_dofs_,
         src.max_jac_entries_};
}

} // namespace assembly
} // namespace femx
