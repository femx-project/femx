#pragma once

#include <algorithm>

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/common/Checks.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/SystemMatrix.hpp>
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
  Index                         num_nodes{0}; ///< Number of mesh nodes on this element.
  VectorView<Space, const Real> state;        ///< State in local degree-of-freedom order.
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

  /**
   * @brief Return one local history state without copying it.
   *
   * @param[in] lag - History lag.
   * @return Local history state.
   */
  FEMX_HOST_DEVICE VectorView<Space, const Real> histState(Index lag) const
  {
    return hist.subview(lag * nxt.size(), nxt.size());
  }
};

using HostElementView       = ElementView<MemorySpace::Host>;
using DeviceElementView     = ElementView<MemorySpace::Device>;
using HostTimeElementView   = TimeElementView<MemorySpace::Host>;
using DeviceTimeElementView = TimeElementView<MemorySpace::Device>;

/// @cond INTERNAL
namespace detail
{
struct CpuWork
{
  HostVector<Real>  state;
  HostVector<Real>  coords;
  HostVector<Real>  hist;
  HostVector<Real>  nxt;
  HostVector<Real>  res;
  DenseMatrix       mat;
  HostVector<Index> rows;
  HostVector<Index> cols;
};

inline CpuWork& cpuWork()
{
  static thread_local CpuWork work;
  return work;
}

inline void checkAssemblyAliases(const HostVector<Real>& state,
                                 const HostVector<Real>& res)
{
  require(&state != &res,
          "Assembly state and residual must not alias");
}

inline void checkAssemblyInputs(const fem::Mesh&        mesh,
                                const HostAssemblyMap&  map,
                                const HostVector<Real>& state)
{
  require(mesh.numElems() == map.numElems(),
          "Mesh and AssemblyMap have different element counts");
  require(state.size() == map.numStates(),
          "Assembly state size does not match AssemblyMap");
}

inline void checkTimeAssemblyInputs(
    Index                      num_hist,
    state::VariableBlock       wrt,
    const HostAssemblyMap&     map,
    HostVectorView<const Real> hist,
    HostVectorView<const Real> nxt)
{
  require(num_hist > 0 && hist.size() == num_hist * map.numStates()
              && nxt.size() == map.numStates(),
          "Time assembly state dimensions do not match AssemblyMap");
  require(!wrt.isParam()
              && (!wrt.isHistoryState() || (wrt.historyLag() >= 0 && wrt.historyLag() < num_hist)),
          "Time assembly variable block is invalid");
}

inline void checkElementRange(
    const HostAssemblyMap& map,
    Index                  elem_begin,
    Index                  elem_end)
{
  require(elem_begin >= 0 && elem_end >= elem_begin
              && elem_end <= map.numElems(),
          "Time assembly element range is invalid");
}

inline void checkTimeAssemblyAliases(HostVectorView<const Real> hist,
                                     HostVectorView<const Real> nxt,
                                     const HostVector<Real>&    res)
{
  require(!femx::detail::overlaps(hist, res.view())
              && !femx::detail::overlaps(nxt, res.view()),
          "Time assembly residual must not alias its inputs");
}

inline void addTimeElement(
    const HostAssemblyMap&                   map,
    Index                                    ie,
    const DenseMatrix&                       elem_mat,
    HostVector<Index>&                       rows,
    HostVector<Index>&                       cols,
    linalg::SystemMatrix<MemorySpace::Host>& jac)
{
  const auto map_v = map.view();
  rows.resize(map_v.numResDofs(ie));
  cols.resize(map_v.numStateDofs(ie));

  for (Index row = 0; row < rows.size(); ++row)
  {
    rows[row] = map_v.resDof(ie, row);
  }

  for (Index col = 0; col < cols.size(); ++col)
  {
    cols[col] = map_v.stateDof(ie, col);
  }

  const Index num_entries = rows.size() * cols.size();
  jac.addElement(
      {rows.view(),
       cols.view(),
       {map_v.jac_map + map_v.jac_offsets[ie], num_entries},
       elem_mat.view()});
}

inline void reduceTimeResidual(HostVector<Real>& res,
                               Index,
                               Index,
                               Index,
                               linalg::Context<MemorySpace::Host>& ctx)
{
  ctx.allReduceSum(res.view());
}

template <class ElementKernel>
void assembleHostElements(
    const ElementKernel&                     kernel,
    const fem::Mesh&                         mesh,
    const HostAssemblyMap&                   map,
    const HostVector<Real>&                  state,
    HostVector<Real>*                        res,
    linalg::SystemMatrix<MemorySpace::Host>* jac,
    linalg::Context<MemorySpace::Host>&      ctx)
{
  checkAssemblyInputs(mesh, map, state);
  require(res != nullptr || jac != nullptr,
          "Assembly requires a residual or Jacobian output");

  if (res != nullptr)
  {
    checkAssemblyAliases(state, *res);
    ctx.vectorHandler().resizeOrZero(*res, map.numRes());
  }

  const auto mesh_v = mesh.view();
  const auto map_v  = map.view();
  const auto range  = ctx.elementRange(map.numElems());

#pragma omp parallel
  {
    auto&             work     = cpuWork();
    HostVector<Real>& state_e  = work.state;
    HostVector<Real>& coords_e = work.coords;
    HostVector<Real>& res_e    = work.res;
    DenseMatrix&      jac_e    = work.mat;
    state_e.reserve(map.maxState());
    coords_e.reserve(mesh.maxElemNodes() * mesh.dim());
    res_e.reserve(map.maxRes());

#pragma omp for
    for (Index ie = range.begin; ie < range.end; ++ie)
    {
      const Index num_rows  = map_v.numResDofs(ie);
      const Index num_cols  = map_v.numStateDofs(ie);
      const Index num_nodes = mesh_v.elemNumNodes(ie);

      state_e.resize(num_cols);
      coords_e.resize(num_nodes * mesh.dim());
      res_e.resize(num_rows);
      jac_e.resize(num_rows, num_cols);

      for (Index col = 0; col < num_cols; ++col)
      {
        state_e[col] = state[map_v.stateDof(ie, col)];
      }
      for (Index in = 0; in < num_nodes; ++in)
      {
        const Index node = mesh_v.elemNode(ie, in);
        for (Index id = 0; id < mesh.dim(); ++id)
        {
          coords_e[in * mesh.dim() + id] = mesh_v.coord(node, id);
        }
      }

      const HostElementView elem{
          ie, mesh.dim(), num_nodes, state_e.view(), coords_e.view()};
      for (Index row = 0; row < num_rows; ++row)
      {
        HostVectorView<Real> jac_row(
            jac_e.data() + row * num_cols, num_cols);
        kernel.evalRow(elem, row, res_e[row], jac_row);
        if (res != nullptr)
        {
#pragma omp atomic update
          (*res)[map_v.resDof(ie, row)] += res_e[row];
        }
      }
      if (jac != nullptr)
      {
        addTimeElement(
            map, ie, jac_e, work.rows, work.cols, *jac);
      }
    }
  }
  if (res != nullptr)
  {
    ctx.allReduceSum(res->view());
  }
}

} // namespace detail

/// @endcond

/**
 * @brief Assemble residual and Jacobian on the CPU reference path.
 *
 * @param[in] kernel - Element evaluator.
 * @param[in] mesh - Host mesh matching the map's element order.
 * @param[in] map - Element-to-global assembly map.
 * @param[in] state - Global state vector.
 * @param[out] res - Global residual replaced by the assembled result.
 * @param[in,out] jac - Jacobian receiving element contributions.
 * @param[in,out] ctx - Host execution context.
 */
template <class ElementKernel>
void assembleResidualAndJacobian(
    const ElementKernel&                     kernel,
    const fem::Mesh&                         mesh,
    const HostAssemblyMap&                   map,
    const HostVector<Real>&                  state,
    HostVector<Real>&                        res,
    linalg::SystemMatrix<MemorySpace::Host>& jac,
    linalg::Context<MemorySpace::Host>&      ctx)
{
  detail::assembleHostElements(
      kernel, mesh, map, state, &res, &jac, ctx);
}

/**
 * @brief Assemble a stationary Host residual without a Jacobian.
 *
 * @param[in] kernel - Element evaluator.
 * @param[in] mesh - Host mesh matching the map's element order.
 * @param[in] map - Element-to-global assembly map.
 * @param[in] state - Global state vector.
 * @param[out] res - Global residual replaced by the assembled result.
 * @param[in,out] ctx - Host execution context.
 */
template <class ElementKernel>
void assembleResidual(const ElementKernel&                kernel,
                      const fem::Mesh&                    mesh,
                      const HostAssemblyMap&              map,
                      const HostVector<Real>&             state,
                      HostVector<Real>&                   res,
                      linalg::Context<MemorySpace::Host>& ctx)
{
  detail::assembleHostElements(
      kernel, mesh, map, state, &res, nullptr, ctx);
}

/**
 * @brief Assemble a stationary Host Jacobian without a residual.
 *
 * @param[in] kernel - Element evaluator.
 * @param[in] mesh - Host mesh matching the map's element order.
 * @param[in] map - Element-to-global assembly map.
 * @param[in] state - Global state vector.
 * @param[in,out] jacobian - Jacobian receiving element contributions.
 * @param[in,out] ctx - Host execution context.
 */
template <class ElementKernel>
void assembleJacobian(
    const ElementKernel&                     kernel,
    const fem::Mesh&                         mesh,
    const HostAssemblyMap&                   map,
    const HostVector<Real>&                  state,
    linalg::SystemMatrix<MemorySpace::Host>& jacobian,
    linalg::Context<MemorySpace::Host>&      ctx)
{
  detail::assembleHostElements(
      kernel, mesh, map, state, nullptr, &jacobian, ctx);
}

/**
 * @brief Assemble a time residual and state Jacobian over an element range.
 *
 * @param[in] kernel - Element evaluator.
 * @param[in] step - Residual step index.
 * @param[in] num_hist - Number of history states.
 * @param[in] wrt - State block differentiated by the Jacobian.
 * @param[in] map - Element-to-global assembly map.
 * @param[in] elem_begin - First element to assemble.
 * @param[in] elem_end - One past the last element to assemble.
 * @param[in] hist - Global lag-major history states.
 * @param[in] nxt - Global next state.
 * @param[out] res - Global residual replaced by the assembled result.
 * @param[in,out] jac - Matrix zeroed and assembled in place.
 * @param[in,out] ctx - Execution context matching `jac`.
 * @throws std::runtime_error - If dimensions, the range, matrix layout, or
 * aliasing are invalid, or if the implementation reports an error.
 */
template <class ElementKernel>
void assembleResidualAndJacobian(
    const ElementKernel&                     kernel,
    Index                                    step,
    Index                                    num_hist,
    state::VariableBlock                     wrt,
    const HostAssemblyMap&                   map,
    Index                                    elem_begin,
    Index                                    elem_end,
    HostVectorView<const Real>               hist,
    HostVectorView<const Real>               nxt,
    HostVector<Real>&                        res,
    linalg::SystemMatrix<MemorySpace::Host>& jac,
    linalg::Context<MemorySpace::Host>&      ctx)
{
  detail::checkTimeAssemblyInputs(num_hist, wrt, map, hist, nxt);
  detail::checkElementRange(map, elem_begin, elem_end);
  detail::checkTimeAssemblyAliases(hist, nxt, res);

  ctx.vectorHandler().resizeOrZero(res, map.numRes());

  const auto map_v = map.view();

#pragma omp parallel
  {
    auto& work = detail::cpuWork();
    work.hist.reserve(num_hist * map.maxState());
    work.nxt.reserve(map.maxState());

#pragma omp for
    for (Index ie = elem_begin; ie < elem_end; ++ie)
    {
      const Index num_rows = map_v.numResDofs(ie);
      const Index num_cols = map_v.numStateDofs(ie);
      work.hist.resize(num_hist * num_cols);
      work.nxt.resize(num_cols);
      work.mat.resize(num_rows, num_cols);

      for (Index lag = 0; lag < num_hist; ++lag)
      {
        for (Index col = 0; col < num_cols; ++col)
        {
          const Index dof                 = map_v.stateDof(ie, col);
          work.hist[lag * num_cols + col] = hist[lag * map.numStates() + dof];
        }
      }
      for (Index col = 0; col < num_cols; ++col)
      {
        work.nxt[col] = nxt[map_v.stateDof(ie, col)];
      }

      const HostTimeElementView elem{
          ie, step, num_hist, work.hist.view(), work.nxt.view()};
      for (Index row = 0; row < num_rows; ++row)
      {
        Real local_res = 0.0;
        kernel.evalRow(elem,
                       wrt,
                       row,
                       local_res,
                       {work.mat.data() + row * num_cols, num_cols});

#pragma omp atomic update
        res[map_v.resDof(ie, row)] += local_res;
      }
      detail::addTimeElement(
          map, ie, work.mat, work.rows, work.cols, jac);
    }
  }
  detail::reduceTimeResidual(
      res, elem_begin, elem_end, map.numElems(), ctx);
}

/**
 * @brief Assemble a time residual over an element range.
 *
 * @param[in] kernel - Element evaluator.
 * @param[in] step - Residual step index.
 * @param[in] num_hist - Number of history states.
 * @param[in] map - Element-to-global assembly map.
 * @param[in] elem_begin - First element to assemble.
 * @param[in] elem_end - One past the last element to assemble.
 * @param[in] hist - Global lag-major history states.
 * @param[in] nxt - Global next state.
 * @param[out] res - Global residual replaced by the assembled result.
 * @param[in,out] ctx - Execution context.
 * @throws std::runtime_error - If dimensions, the range, or aliasing are
 * invalid, or if the implementation reports an error.
 */
template <class ElementKernel, class Context>
void assembleResidual(
    const ElementKernel&       kernel,
    Index                      step,
    Index                      num_hist,
    const HostAssemblyMap&     map,
    Index                      elem_begin,
    Index                      elem_end,
    HostVectorView<const Real> hist,
    HostVectorView<const Real> nxt,
    HostVector<Real>&          res,
    Context&                   ctx)
{
  detail::checkTimeAssemblyInputs(
      num_hist, state::VariableBlock::NextState, map, hist, nxt);
  detail::checkElementRange(map, elem_begin, elem_end);
  detail::checkTimeAssemblyAliases(hist, nxt, res);
  ctx.vectorHandler().resizeOrZero(res, map.numRes());

  const auto map_v = map.view();
#pragma omp parallel
  {
    auto& work = detail::cpuWork();
    work.hist.reserve(num_hist * map.maxState());
    work.nxt.reserve(map.maxState());

#pragma omp for
    for (Index ie = elem_begin; ie < elem_end; ++ie)
    {
      const Index num_rows = map_v.numResDofs(ie);
      const Index num_cols = map_v.numStateDofs(ie);
      work.hist.resize(num_hist * num_cols);
      work.nxt.resize(num_cols);

      for (Index lag = 0; lag < num_hist; ++lag)
      {
        for (Index col = 0; col < num_cols; ++col)
        {
          const Index dof                 = map_v.stateDof(ie, col);
          work.hist[lag * num_cols + col] = hist[lag * map.numStates() + dof];
        }
      }
      for (Index col = 0; col < num_cols; ++col)
      {
        work.nxt[col] = nxt[map_v.stateDof(ie, col)];
      }

      const HostTimeElementView elem{
          ie, step, num_hist, work.hist.view(), work.nxt.view()};
      for (Index row = 0; row < num_rows; ++row)
      {
        Real local_res = 0.0;
        kernel.evalRow(elem,
                       state::VariableBlock::NextState,
                       row,
                       local_res,
                       HostVectorView<Real>{});
#pragma omp atomic update
        res[map_v.resDof(ie, row)] += local_res;
      }
    }
  }
  detail::reduceTimeResidual(
      res, elem_begin, elem_end, map.numElems(), ctx);
}

} // namespace assembly
} // namespace femx
