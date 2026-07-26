#pragma once

#include <utility>

#include <femx/common/Types.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>

namespace femx
{
namespace fem
{
class DofMap;
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
  const Index* res_dofs{nullptr};      ///< Element residual-to-global degrees of freedom.
  const Index* state_offsets{nullptr}; ///< Element offsets into state_dofs.
  const Index* state_dofs{nullptr};    ///< Element state-to-global degrees of freedom.
  const Index* jac_offsets{nullptr};   ///< Element offsets into jac_map.
  const Index* jac_map{nullptr};       ///< Local Jacobian-to-CSR entries.

  Index max_res{0};   ///< Maximum residual degrees of freedom on one element.
  Index max_state{0}; ///< Maximum state degrees of freedom on one element.
  Index max_jac{0};   ///< Maximum local Jacobian entries.

  /**
   * @brief Return the number of residual degrees of freedom on an element.
   *
   * @param[in] ie - Element index.
   * @return Number of residual degrees of freedom.
   */
  FEMX_HOST_DEVICE Index numResDofs(Index ie) const
  {
    return res_offsets[ie + 1] - res_offsets[ie];
  }

  /**
   * @brief Return the number of state degrees of freedom on an element.
   *
   * @param[in] ie - Element index.
   * @return Number of state degrees of freedom.
   */
  FEMX_HOST_DEVICE Index numStateDofs(Index ie) const
  {
    return state_offsets[ie + 1] - state_offsets[ie];
  }

  /**
   * @brief Map an element residual row to a global residual degree of freedom.
   *
   * @param[in] ie - Element index.
   * @param[in] i - Element-local residual row.
   * @return Global residual degree of freedom.
   */
  FEMX_HOST_DEVICE Index resDof(Index ie, Index i) const
  {
    return res_dofs[res_offsets[ie] + i];
  }

  /**
   * @brief Map an element state column to a global state degree of freedom.
   *
   * @param[in] ie - Element index.
   * @param[in] i - Element-local state column.
   * @return Global state degree of freedom.
   */
  FEMX_HOST_DEVICE Index stateDof(Index ie, Index i) const
  {
    return state_dofs[state_offsets[ie] + i];
  }

  /**
   * @brief Map a local Jacobian entry to a CSR value index.
   *
   * @param[in] ie - Element index.
   * @param[in] i - Row-major local Jacobian index.
   * @return CSR value index.
   */
  FEMX_HOST_DEVICE Index jacIndex(Index ie, Index i) const
  {
    return jac_map[jac_offsets[ie] + i];
  }
};

using HostAssemblyMapView   = AssemblyMapView<MemorySpace::Host>;
using DeviceAssemblyMapView = AssemblyMapView<MemorySpace::Device>;

template <MemorySpace Space>
class AssemblyMap;

using HostAssemblyMap   = AssemblyMap<MemorySpace::Host>;
using DeviceAssemblyMap = AssemblyMap<MemorySpace::Device>;

/**
 * @brief Own runtime element degrees of freedom and local-to-CSR mappings.
 *
 * The map is built once on the host. Its flat arrays can then be copied once
 * to device memory without introducing compile-time element traits.
 */
template <MemorySpace Space>
class AssemblyMap
{
public:
  AssemblyMap() = default;

  AssemblyMap(const AssemblyMap&)                = default;
  AssemblyMap(AssemblyMap&&) noexcept            = default;
  AssemblyMap& operator=(const AssemblyMap&)     = default;
  AssemblyMap& operator=(AssemblyMap&&) noexcept = default;

private:
  AssemblyMap(Index                num_elems,
              Index                num_res,
              Index                num_states,
              Vector<Space, Index> res_offsets,
              Vector<Space, Index> res_dofs,
              Vector<Space, Index> state_offsets,
              Vector<Space, Index> state_dofs,
              Vector<Space, Index> jac_offsets,
              Vector<Space, Index> jac_map,
              CsrPattern<Space>    pattern,
              Index                max_res,
              Index                max_state,
              Index                max_jac)
    : num_elems_(num_elems),
      num_res_(num_res),
      num_states_(num_states),
      res_offsets_(std::move(res_offsets)),
      res_dofs_(std::move(res_dofs)),
      state_offsets_(std::move(state_offsets)),
      state_dofs_(std::move(state_dofs)),
      jac_offsets_(std::move(jac_offsets)),
      jac_map_(std::move(jac_map)),
      pattern_(std::move(pattern)),
      max_res_(max_res),
      max_state_(max_state),
      max_jac_(max_jac)
  {
  }

  friend HostAssemblyMap makeAssemblyMap(
      Index                                num_res,
      Index                                num_states,
      const HostVector<HostVector<Index>>& res_dofs,
      const HostVector<HostVector<Index>>& state_dofs);

  friend void copy(const HostAssemblyMap& src,
                   DeviceAssemblyMap&     dst,
                   linalg::CudaContext&   ctx);

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

  /** @brief Return the immutable global Jacobian sparsity pattern. */
  const CsrPattern<Space>& pattern() const noexcept
  {
    return pattern_;
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
  Index num_elems_{0};  ///< Number of mapped elements.
  Index num_res_{0};    ///< Global residual size.
  Index num_states_{0}; ///< Global state size.

  Vector<Space, Index> res_offsets_;   ///< Residual degree-of-freedom offsets.
  Vector<Space, Index> res_dofs_;      ///< Residual degrees of freedom.
  Vector<Space, Index> state_offsets_; ///< State degree-of-freedom offsets.
  Vector<Space, Index> state_dofs_;    ///< State degrees of freedom.
  Vector<Space, Index> jac_offsets_;   ///< Local Jacobian offsets.
  Vector<Space, Index> jac_map_;       ///< Local Jacobian-to-CSR mapping.
  CsrPattern<Space>    pattern_;       ///< Immutable CSR sparsity pattern.

  Index max_res_{0};   ///< Largest element residual size.
  Index max_state_{0}; ///< Largest element state size.
  Index max_jac_{0};   ///< Largest element Jacobian size.
};

/**
 * @brief Build a host assembly map from explicit element DOF tables.
 *
 * @param[in] num_res - Global residual size.
 * @param[in] num_states - Global state size.
 * @param[in] res_dofs - Residual degrees of freedom for each element.
 * @param[in] state_dofs - State degrees of freedom for each element.
 * @return Validated map and its immutable CSR pattern.
 */
HostAssemblyMap makeAssemblyMap(
    Index                                num_res,
    Index                                num_states,
    const HostVector<HostVector<Index>>& res_dofs,
    const HostVector<HostVector<Index>>& state_dofs);

/**
 * @brief Build a rectangular assembly map from residual and state maps.
 *
 * @param[in] res_map - Element-to-global map for residual rows.
 * @param[in] state_map - Element-to-global map for state columns.
 * @return Validated assembly map and its immutable sparse pattern.
 */
HostAssemblyMap makeAssemblyMap(const fem::DofMap& res_map,
                                const fem::DofMap& state_map);

/**
 * @brief Build a square assembly map using one degree-of-freedom map.
 *
 * @param[in] dof_map - Element-to-global map for rows and columns.
 * @return Validated assembly map and its immutable sparse pattern.
 */
HostAssemblyMap makeAssemblyMap(const fem::DofMap& dof_map);

/**
 * @brief Copy a host assembly map and pattern to device-owned storage.
 *
 * The copy is enqueued on `ctx`; keep `src` alive until earlier queued reads
 * have completed.
 *
 * @param[in] src - Host assembly map.
 * @param[out] dst - Device assembly map.
 * @param[in,out] ctx - CUDA context receiving the copies.
 */
void copy(const HostAssemblyMap& src,
          DeviceAssemblyMap&     dst,
          linalg::CudaContext&   ctx);

} // namespace assembly
} // namespace femx
