#pragma once

#include <utility>

#include <femx/common/Context.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/CsrGraph.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace fem
{
class DofLayout;
}

namespace assembly
{

/** @brief Non-owning assembly mapping consumed by host and device kernels. */
template <MemorySpace Space>
struct AssemblyMapView
{
  Index num_elems{0};
  Index num_res{0};
  Index num_states{0};

  const Index* res_offsets{nullptr};
  const Index* res_dofs{nullptr};
  const Index* state_offsets{nullptr};
  const Index* state_dofs{nullptr};
  const Index* jac_offsets{nullptr};
  const Index* jac_map{nullptr};

  Index max_res_dofs{0};
  Index max_state_dofs{0};
  Index max_jac_entries{0};

  FEMX_HOST_DEVICE Index numResDofs(Index ie) const
  {
    return res_offsets[ie + 1] - res_offsets[ie];
  }

  FEMX_HOST_DEVICE Index numStateDofs(Index ie) const
  {
    return state_offsets[ie + 1] - state_offsets[ie];
  }

  FEMX_HOST_DEVICE Index resDof(Index ie, Index i) const
  {
    return res_dofs[res_offsets[ie] + i];
  }

  FEMX_HOST_DEVICE Index stateDof(Index ie, Index i) const
  {
    return state_dofs[state_offsets[ie] + i];
  }

  FEMX_HOST_DEVICE Index jacIndex(Index ie, Index i) const
  {
    return jac_map[jac_offsets[ie] + i];
  }
};

/**
 * @brief Runtime element DOFs and local-entry-to-CSR mappings.
 *
 * The map is built once on the host. Its flat arrays can then be copied once
 * to device memory without introducing compile-time element traits.
 */
template <MemorySpace Space>
class AssemblyMap
{
public:
  using IndexVector = Vector<Space, Index>;

  AssemblyMap() = default;

  AssemblyMap(const AssemblyMap&)                = default;
  AssemblyMap(AssemblyMap&&) noexcept            = default;
  AssemblyMap& operator=(const AssemblyMap&)     = default;
  AssemblyMap& operator=(AssemblyMap&&) noexcept = default;

private:
  AssemblyMap(Index           num_elems,
              Index           num_res,
              Index           num_states,
              IndexVector     res_offsets,
              IndexVector     res_dofs,
              IndexVector     state_offsets,
              IndexVector     state_dofs,
              IndexVector     jac_offsets,
              IndexVector     jac_map,
              CsrGraph<Space> graph,
              Index           max_res_dofs,
              Index           max_state_dofs,
              Index           max_jac_entries)
    : num_elems_(num_elems),
      num_res_(num_res),
      num_states_(num_states),
      res_offsets_(std::move(res_offsets)),
      res_dofs_(std::move(res_dofs)),
      state_offsets_(std::move(state_offsets)),
      state_dofs_(std::move(state_dofs)),
      jac_offsets_(std::move(jac_offsets)),
      jac_map_(std::move(jac_map)),
      graph_(std::move(graph)),
      max_res_dofs_(max_res_dofs),
      max_state_dofs_(max_state_dofs),
      max_jac_entries_(max_jac_entries)
  {
  }

  friend AssemblyMap<MemorySpace::Host> makeAssemblyMap(
      Index,
      Index,
      const Array<Array<Index>>&,
      const Array<Array<Index>>&);

  friend void copy(const AssemblyMap<MemorySpace::Host>&,
                   AssemblyMap<MemorySpace::Device>&,
                   CudaContext&);

public:
  Index numElems() const noexcept
  {
    return num_elems_;
  }

  Index numRes() const noexcept
  {
    return num_res_;
  }

  Index numStates() const noexcept
  {
    return num_states_;
  }

  Index maxResDofs() const noexcept
  {
    return max_res_dofs_;
  }

  Index maxStateDofs() const noexcept
  {
    return max_state_dofs_;
  }

  Index maxJacEntries() const noexcept
  {
    return max_jac_entries_;
  }

  const CsrGraph<Space>& graph() const noexcept
  {
    return graph_;
  }

  AssemblyMapView<Space> view() const noexcept
  {
    return {num_elems_,
            num_res_,
            num_states_,
            res_offsets_.data(),
            res_dofs_.data(),
            state_offsets_.data(),
            state_dofs_.data(),
            jac_offsets_.data(),
            jac_map_.data(),
            max_res_dofs_,
            max_state_dofs_,
            max_jac_entries_};
  }

private:
  Index num_elems_{0};
  Index num_res_{0};
  Index num_states_{0};

  IndexVector     res_offsets_;
  IndexVector     res_dofs_;
  IndexVector     state_offsets_;
  IndexVector     state_dofs_;
  IndexVector     jac_offsets_;
  IndexVector     jac_map_;
  CsrGraph<Space> graph_;

  Index max_res_dofs_{0};
  Index max_state_dofs_{0};
  Index max_jac_entries_{0};
};

using HostAssemblyMap   = AssemblyMap<MemorySpace::Host>;
using DeviceAssemblyMap = AssemblyMap<MemorySpace::Device>;

AssemblyMap<MemorySpace::Host> makeAssemblyMap(
    Index                      num_res,
    Index                      num_states,
    const Array<Array<Index>>& res_dofs,
    const Array<Array<Index>>& state_dofs);

AssemblyMap<MemorySpace::Host> makeAssemblyMap(
    fem::DofLayout res_lyt,
    fem::DofLayout state_lyt);

AssemblyMap<MemorySpace::Host> makeAssemblyMap(fem::DofLayout layout);

void copy(const AssemblyMap<MemorySpace::Host>& src,
          AssemblyMap<MemorySpace::Device>&     dst,
          CudaContext&                          ctx);

} // namespace assembly
} // namespace femx
