#pragma once

#include <stdexcept>

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/common/Context.hpp>
#include <femx/fem/Geometry.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/TimeResidual.hpp>

namespace femx
{
namespace assembly
{

/** @brief Runtime element inputs shared by CPU and CUDA operators. */
template <MemorySpace Space>
struct ElementView
{
  Index                         ie{0};        ///< Global element index.
  Index                         dim{0};       ///< Spatial dimension.
  Index                         num_nodes{0}; ///< Number of geometry nodes on this element.
  VectorView<Space, const Real> state;        ///< Element state in local DOF order.
  VectorView<Space, const Real> coords;       ///< Node-major element coordinates.
};

/** @brief Element-local inputs for one time-dependent residual step. */
template <MemorySpace Space>
struct TimeElementView
{
  Index                         ie{0};       ///< Global element index.
  Index                         step{0};     ///< Residual step index.
  Index                         num_hist{0}; ///< Number of local history states.
  VectorView<Space, const Real> hist;        ///< History states, lag-major.
  VectorView<Space, const Real> nxt;         ///< Local next state.

  /** @brief Return one local history state without copying it. */
  FEMX_HOST_DEVICE VectorView<Space, const Real> histState(Index lag) const
  {
    return hist.subview(lag * nxt.size(), nxt.size());
  }
};

/// @cond INTERNAL
namespace detail
{
struct CpuWork
{
  HostVector state;
  HostVector coords;
  HostVector hist;
  HostVector nxt;
  HostVector res;
  HostVector jac;
};

inline CpuWork& cpuWork()
{
  static thread_local CpuWork work;
  return work;
}

template <MemorySpace Space>
void checkAssemblyAliases(const Vector<Space>& state,
                          const Vector<Space>& res,
                          const Vector<Space>& vals)
{
  if (&state == &res || &state == &vals || &res == &vals)
  {
    throw std::runtime_error(
        "Assembly state, residual, and matrix values must not alias");
  }
}

inline void checkAssemblyInputs(const fem::HostGeometry&              geom,
                                const AssemblyMap<MemorySpace::Host>& map,
                                const HostVector&                     state,
                                const HostCsrMatrix&                  jac)
{
  if (geom.numElems() != map.numElems())
  {
    throw std::runtime_error(
        "Geometry and AssemblyMap have different element counts");
  }
  if (state.size() != map.numStates())
  {
    throw std::runtime_error(
        "Assembly state size does not match AssemblyMap");
  }
  if (jac.graph().layoutId() != map.graph().layoutId())
  {
    throw std::runtime_error(
        "Assembly matrix must use the AssemblyMap CSR layout");
  }
}

inline void checkTimeAssemblyInputs(
    Index                                 num_hist,
    state::VariableBlock                  wrt,
    const AssemblyMap<MemorySpace::Host>& map,
    const HostVector&                     hist,
    const HostVector&                     nxt,
    const HostCsrMatrix&                  jac)
{
  if (num_hist <= 0 || hist.size() != num_hist * map.numStates()
      || nxt.size() != map.numStates())
  {
    throw std::runtime_error(
        "Time assembly state dimensions do not match AssemblyMap");
  }
  if (wrt.isParam()
      || (wrt.isHistoryState()
          && (wrt.historyLag() < 0 || wrt.historyLag() >= num_hist)))
  {
    throw std::runtime_error("Time assembly variable block is invalid");
  }
  if (jac.graph().layoutId() != map.graph().layoutId())
  {
    throw std::runtime_error(
        "Time assembly matrix must use the AssemblyMap CSR layout");
  }
}

template <MemorySpace Space>
void checkTimeAssemblyAliases(const Vector<Space>& hist,
                              const Vector<Space>& nxt,
                              const Vector<Space>& res,
                              const Vector<Space>& vals)
{
  if (&hist == &res || &hist == &vals || &nxt == &res || &nxt == &vals
      || &res == &vals)
  {
    throw std::runtime_error(
        "Time assembly outputs must not alias inputs or each other");
  }
}
} // namespace detail

/// @endcond

/**
 * @brief Assemble residual and Jacobian on the CPU reference path.
 *
 * ElementOperator is the only template parameter. It implements
 * `evalRow(ElementView<Host>, local_row, res, jac_row)` and receives
 * runtime element sizes through the views.
 *
 * @tparam ElementOperator Row-wise element residual and Jacobian evaluator.
 * @param op Element evaluator.
 * @param geom Host geometry matching the map's element order.
 * @param map Element-to-global assembly map.
 * @param state Global state vector.
 * @param res Global residual replaced by the assembled result.
 * @param jac CSR matrix zeroed and assembled in place.
 */
template <class ElementOperator>
void assemble(const ElementOperator&                op,
              const fem::HostGeometry&              geom,
              const AssemblyMap<MemorySpace::Host>& map,
              const HostVector&                     state,
              HostVector&                           res,
              HostCsrMatrix&                        jac,
              CpuContext&)
{
  detail::checkAssemblyInputs(geom, map, state, jac);
  const HostVector& mat_vals = jac.vals();
  detail::checkAssemblyAliases(state, res, mat_vals);

  resizeOrZero(res, map.numRes());
  jac.setZero();

  const auto geom_v = geom.view();
  const auto map_v  = map.view();

  auto&       work     = detail::cpuWork();
  HostVector& state_e  = work.state;
  HostVector& coords_e = work.coords;
  HostVector& res_e    = work.res;
  HostVector& jac_e    = work.jac;
  state_e.reserve(map.maxState());
  coords_e.reserve(geom.maxElemNodes() * geom.dim());
  res_e.reserve(map.maxRes());
  jac_e.reserve(map.maxJac());

  for (Index ie = 0; ie < map.numElems(); ++ie)
  {
    const Index num_rows  = map_v.numResDofs(ie);
    const Index num_cols  = map_v.numStateDofs(ie);
    const Index num_nodes = geom_v.elemNumNodes(ie);

    state_e.resize(num_cols);
    coords_e.resize(num_nodes * geom.dim());
    res_e.resize(num_rows);
    jac_e.resize(num_rows * num_cols);

    for (Index col = 0; col < num_cols; ++col)
    {
      state_e[col] = state[map_v.stateDof(ie, col)];
    }
    for (Index in = 0; in < num_nodes; ++in)
    {
      const Index node = geom_v.elemNode(ie, in);
      for (Index d = 0; d < geom.dim(); ++d)
      {
        coords_e[in * geom.dim() + d] = geom_v.coord(node, d);
      }
    }

    const ElementView<MemorySpace::Host> elem{
        ie, geom.dim(), num_nodes, state_e.view(), coords_e.view()};

    for (Index row = 0; row < num_rows; ++row)
    {
      HostVectorView jac_row(jac_e.data() + row * num_cols, num_cols);
      op.evalRow(elem, row, res_e[row], jac_row);
    }

    for (Index row = 0; row < num_rows; ++row)
    {
      res[map_v.resDof(ie, row)] += res_e[row];
    }
    for (Index i = 0; i < num_rows * num_cols; ++i)
    {
      jac.valsData()[map_v.jacIndex(ie, i)] += jac_e[i];
    }
  }
}

/**
 * @brief Assemble one time residual and state Jacobian on the CPU.
 *
 * The operator implements
 * `evalRow(TimeElementView<Host>, wrt, row, res, jac_row)`. History storage is
 * lag-major with `map.numStates()` global values per lag.
 */
template <class ElementOperator>
void assemble(const ElementOperator&                op,
              Index                                 step,
              Index                                 num_hist,
              state::VariableBlock                  wrt,
              const AssemblyMap<MemorySpace::Host>& map,
              const HostVector&                     hist,
              const HostVector&                     nxt,
              HostVector&                           res,
              HostCsrMatrix&                        jac,
              CpuContext&)
{
  detail::checkTimeAssemblyInputs(num_hist, wrt, map, hist, nxt, jac);
  const HostVector& vals = jac.vals();
  detail::checkTimeAssemblyAliases(hist, nxt, res, vals);

  resizeOrZero(res, map.numRes());
  jac.setZero();

  const auto map_v = map.view();
  auto&      work  = detail::cpuWork();
  work.hist.reserve(num_hist * map.maxState());
  work.nxt.reserve(map.maxState());
  work.res.reserve(map.maxRes());
  work.jac.reserve(map.maxJac());

  for (Index ie = 0; ie < map.numElems(); ++ie)
  {
    const Index nrow = map_v.numResDofs(ie);
    const Index ncol = map_v.numStateDofs(ie);
    work.hist.resize(num_hist * ncol);
    work.nxt.resize(ncol);
    work.res.resize(nrow);
    work.jac.resize(nrow * ncol);

    for (Index lag = 0; lag < num_hist; ++lag)
    {
      for (Index col = 0; col < ncol; ++col)
      {
        const Index dof = map_v.stateDof(ie, col);
        work.hist[lag * ncol + col] =
            hist[lag * map.numStates() + dof];
      }
    }
    for (Index col = 0; col < ncol; ++col)
    {
      work.nxt[col] = nxt[map_v.stateDof(ie, col)];
    }

    const TimeElementView<MemorySpace::Host> elem{
        ie, step, num_hist, work.hist.view(), work.nxt.view()};
    for (Index row = 0; row < nrow; ++row)
    {
      HostVectorView jac_row(work.jac.data() + row * ncol, ncol);
      op.evalRow(elem, wrt, row, work.res[row], jac_row);
    }

    for (Index row = 0; row < nrow; ++row)
    {
      res[map_v.resDof(ie, row)] += work.res[row];
    }
    for (Index i = 0; i < nrow * ncol; ++i)
    {
      jac.valsData()[map_v.jacIndex(ie, i)] += work.jac[i];
    }
  }
}

} // namespace assembly
} // namespace femx
