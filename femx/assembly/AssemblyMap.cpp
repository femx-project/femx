#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/common/Checks.hpp>
#include <femx/fem/DofMap.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>

namespace femx
{
namespace assembly
{
namespace
{

Index checkedMul(Index lhs, Index rhs)
{
  require(lhs >= 0 && rhs >= 0,
          "AssemblyMap local dimensions must be non-negative");
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

void checkDofs(const HostVector<Index>& dofs, Index size, const char* kind)
{
  for (Index dof : dofs)
  {
    require(dof >= 0 && dof < size,
            std::string("AssemblyMap ") + kind + " DOF is out of range");
  }
}

} // namespace

HostAssemblyMap makeAssemblyMap(
    Index                                num_res,
    Index                                num_states,
    const HostVector<HostVector<Index>>& elem_res,
    const HostVector<HostVector<Index>>& elem_state)
{
  require(num_res >= 0 && num_states >= 0,
          "AssemblyMap global dimensions must be non-negative");
  require(elem_res.size() == elem_state.size(),
          "AssemblyMap residual/state element counts differ");

  const Index num_elem = elem_res.size();

  HostVector<Index> res_offsets(num_elem + 1);
  HostVector<Index> state_offsets(num_elem + 1);
  HostVector<Index> jac_offsets(num_elem + 1);
  HostVector<Index> res_dofs;
  HostVector<Index> state_dofs;

  Index max_res   = 0;
  Index max_state = 0;
  Index max_jac   = 0;

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
    max_res               = std::max(max_res, rows.size());
    max_state             = std::max(max_state, cols.size());
    max_jac               = std::max(max_jac, elem_nnz);
  }

  const Index       nnz = jac_offsets[num_elem];
  HostVector<Index> coo_rows(nnz);
  HostVector<Index> coo_cols(nnz);
  HostVector<Index> order(nnz);
  HostVector<Index> jac_map(nnz);

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

  HostVector<Index> row_ptr(num_res + 1, 0);
  HostVector<Index> cols;
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

  HostCsrPattern pattern(num_res,
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
          std::move(pattern),
          max_res,
          max_state,
          max_jac};
}

HostAssemblyMap makeAssemblyMap(const fem::DofMap& res_map,
                                const fem::DofMap& state_map)
{
  require(res_map.numElems() == state_map.numElems(),
          "AssemblyMap residual/state maps have different element counts");

  HostVector<HostVector<Index>> res_dofs(res_map.numElems());
  HostVector<HostVector<Index>> state_dofs(state_map.numElems());
  for (Index ie = 0; ie < res_map.numElems(); ++ie)
  {
    res_dofs[ie]   = res_map.elementDofs(ie);
    state_dofs[ie] = state_map.elementDofs(ie);
  }

  return makeAssemblyMap(
      res_map.numDofs(), state_map.numDofs(), res_dofs, state_dofs);
}

HostAssemblyMap makeAssemblyMap(const fem::DofMap& dof_map)
{
  return makeAssemblyMap(dof_map, dof_map);
}

void copy(const HostAssemblyMap& src,
          DeviceAssemblyMap&     dst,
          linalg::CudaContext&   ctx)
{
  auto&               vec_handler = ctx.vectors();
  DeviceVector<Index> res_offsets;
  DeviceVector<Index> res_dofs;
  DeviceVector<Index> state_offsets;
  DeviceVector<Index> state_dofs;
  DeviceVector<Index> jac_offsets;
  DeviceVector<Index> jac_map;
  DeviceCsrPattern    pattern;

  vec_handler.copy(src.res_offsets_, res_offsets);
  vec_handler.copy(src.res_dofs_, res_dofs);
  vec_handler.copy(src.state_offsets_, state_offsets);
  vec_handler.copy(src.state_dofs_, state_dofs);
  vec_handler.copy(src.jac_offsets_, jac_offsets);
  vec_handler.copy(src.jac_map_, jac_map);
  femx::copy(src.pattern_, pattern, ctx);

  dst = {src.num_elems_,
         src.num_res_,
         src.num_states_,
         std::move(res_offsets),
         std::move(res_dofs),
         std::move(state_offsets),
         std::move(state_dofs),
         std::move(jac_offsets),
         std::move(jac_map),
         std::move(pattern),
         src.max_res_,
         src.max_state_,
         src.max_jac_};
}

} // namespace assembly
} // namespace femx
