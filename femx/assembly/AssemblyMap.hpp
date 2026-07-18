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
  Index num_elems{0};  ///< Number of elements.
  Index num_res{0};    ///< Global residual size.
  Index num_states{0}; ///< Global state size.

  const Index* res_offsets{nullptr};   ///< Element offsets into res_dofs.
  const Index* res_dofs{nullptr};      ///< Element residual-to-global DOFs.
  const Index* state_offsets{nullptr}; ///< Element offsets into state_dofs.
  const Index* state_dofs{nullptr};    ///< Element state-to-global DOFs.
  const Index* jac_offsets{nullptr};   ///< Element offsets into jac_map.
  const Index* jac_map{nullptr};       ///< Local Jacobian-to-CSR entries.

  Index max_res{0};   ///< Maximum residual DOFs on one element.
  Index max_state{0}; ///< Maximum state DOFs on one element.
  Index max_jac{0};   ///< Maximum local Jacobian entries.

  /** @brief Return the number of residual DOFs on element `ie`. */
  FEMX_HOST_DEVICE Index numResDofs(Index ie) const
  {
    return res_offsets[ie + 1] - res_offsets[ie];
  }

  /** @brief Return the number of state DOFs on element `ie`. */
  FEMX_HOST_DEVICE Index numStateDofs(Index ie) const
  {
    return state_offsets[ie + 1] - state_offsets[ie];
  }

  /** @brief Map an element residual row to a global residual DOF. */
  FEMX_HOST_DEVICE Index resDof(Index ie, Index i) const
  {
    return res_dofs[res_offsets[ie] + i];
  }

  /** @brief Map an element state column to a global state DOF. */
  FEMX_HOST_DEVICE Index stateDof(Index ie, Index i) const
  {
    return state_dofs[state_offsets[ie] + i];
  }

  /** @brief Map a row-major local Jacobian entry to a CSR value index. */
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
  /** @brief Index-vector type in this map's memory space. */
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
              Index           max_res,
              Index           max_state,
              Index           max_jac)
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
      max_res_(max_res),
      max_state_(max_state),
      max_jac_(max_jac)
  {
  }

  friend AssemblyMap<MemorySpace::Host> makeAssemblyMap(
      Index                      num_res,
      Index                      num_states,
      const Array<Array<Index>>& res_dofs,
      const Array<Array<Index>>& state_dofs);

  friend void copy(const AssemblyMap<MemorySpace::Host>& src,
                   AssemblyMap<MemorySpace::Device>&     dst,
                   CudaContext&                          ctx);

public:
  /** @brief Return the number of mapped elements. */
  Index numElems() const noexcept
  {
    return num_elems_;
  }

  /** @brief Return the global residual size. */
  Index numRes() const noexcept
  {
    return num_res_;
  }

  /** @brief Return the global state size. */
  Index numStates() const noexcept
  {
    return num_states_;
  }

  /** @brief Return the largest element residual workspace size. */
  Index maxRes() const noexcept
  {
    return max_res_;
  }

  /** @brief Return the largest element state workspace size. */
  Index maxState() const noexcept
  {
    return max_state_;
  }

  /** @brief Return the largest element Jacobian workspace size. */
  Index maxJac() const noexcept
  {
    return max_jac_;
  }

  /** @brief Return the immutable global Jacobian sparsity graph. */
  const CsrGraph<Space>& graph() const noexcept
  {
    return graph_;
  }

  /** @brief Return a non-owning kernel view valid while this map is alive. */
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
            max_res_,
            max_state_,
            max_jac_};
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

  Index max_res_{0};
  Index max_state_{0};
  Index max_jac_{0};
};

using HostAssemblyMap   = AssemblyMap<MemorySpace::Host>;
using DeviceAssemblyMap = AssemblyMap<MemorySpace::Device>;

/**
 * @brief Build a host assembly map from explicit element DOF tables.
 *
 * @param num_res Global residual size.
 * @param num_states Global state size.
 * @param res_dofs Residual DOFs for each element.
 * @param state_dofs State DOFs for each element.
 * @return Validated map and its immutable CSR graph.
 */
AssemblyMap<MemorySpace::Host> makeAssemblyMap(
    Index                      num_res,
    Index                      num_states,
    const Array<Array<Index>>& res_dofs,
    const Array<Array<Index>>& state_dofs);

/**
 * @brief Build a rectangular assembly map from residual and state layouts.
 * @param res_lyt Element layout for residual rows.
 * @param state_lyt Element layout for state columns.
 */
AssemblyMap<MemorySpace::Host> makeAssemblyMap(
    fem::DofLayout res_lyt,
    fem::DofLayout state_lyt);

/** @brief Build a square assembly map using one layout for rows and columns. */
AssemblyMap<MemorySpace::Host> makeAssemblyMap(fem::DofLayout layout);

/**
 * @brief Copy a host assembly map and graph to device-owned storage.
 *
 * The copy is enqueued on `ctx`; keep `src` alive until earlier queued reads
 * have completed.
 */
void copy(const AssemblyMap<MemorySpace::Host>& src,
          AssemblyMap<MemorySpace::Device>&     dst,
          CudaContext&                          ctx);

} // namespace assembly
} // namespace femx
