#pragma once

#include <algorithm>

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/common/Checks.hpp>
#include <femx/fem/Geometry.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Jacobian.hpp>
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

using HostElementView       = ElementView<MemorySpace::Host>;
using DeviceElementView     = ElementView<MemorySpace::Device>;
using HostTimeElementView   = TimeElementView<MemorySpace::Host>;
using DeviceTimeElementView = TimeElementView<MemorySpace::Device>;

/** @brief Accumulate one row-major element matrix into Host CSR values. */
void addElem(const HostAssemblyMap& map,
             Index                  ie,
             const DenseMatrix&     elem_mat,
             HostCsrMatrix&         mat,
             bool                   atomic = false);

/** @brief Replace selected Host CSR rows by diagonal rows. */
void replaceRows(HostCsrMatrix&           mat,
                 const HostVector<Index>& rows,
                 Real                     diag);

/** @brief Eliminate selected Host CSR columns and correct the right-hand side. */
void eliminateColumns(HostCsrMatrix&           mat,
                      const HostVector<Index>& rows,
                      HostVector<Real>&        rhs);

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
  HostVector<Real>  jac;
  DenseMatrix       mat;
  HostVector<Index> rows;
  HostVector<Index> cols;
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
  require(&state != &res && &state != &vals && &res != &vals,
          "Assembly state, residual, and matrix values must not alias");
}

inline void checkAssemblyInputs(const fem::HostGeometry& geom,
                                const HostAssemblyMap&   map,
                                const HostVector<Real>&  state,
                                const HostCsrMatrix&     jac)
{
  require(geom.numElems() == map.numElems(),
          "Geometry and AssemblyMap have different element counts");
  require(state.size() == map.numStates(),
          "Assembly state size does not match AssemblyMap");
  require(jac.pattern().layoutId() == map.pattern().layoutId(),
          "Assembly matrix must use the AssemblyMap CSR layout");
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
    const HostAssemblyMap&               map,
    Index                                ie,
    const DenseMatrix&                   elem_mat,
    HostVector<Index>&                   rows,
    HostVector<Index>&                   cols,
    linalg::Jacobian<MemorySpace::Host>& jac)
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

} // namespace detail

/// @endcond

/**
 * @brief Assemble residual and Jacobian on the CPU reference path.
 *
 * ElementKernel is the only template parameter. It implements
 * `evalRow(ElementView<Host>, local_row, res, jac_row)` and receives
 * runtime element sizes through the views.
 *
 * @tparam ElementKernel Row-wise element residual and Jacobian evaluator.
 * @param kernel Element evaluator.
 * @param geom Host geometry matching the map's element order.
 * @param map Element-to-global assembly map.
 * @param state Global state vector.
 * @param res Global residual replaced by the assembled result.
 * @param jac CSR matrix zeroed and assembled in place.
 */
template <class ElementKernel>
void assemble(const ElementKernel&                kernel,
              const fem::HostGeometry&            geom,
              const HostAssemblyMap&              map,
              const HostVector<Real>&             state,
              HostVector<Real>&                   res,
              HostCsrMatrix&                      jac,
              linalg::Context<MemorySpace::Host>& ctx)
{
  auto& vec_handler = ctx.vectors();
  detail::checkAssemblyInputs(geom, map, state, jac);
  const HostVector<Real>& mat_vals = jac.vals();
  detail::checkAssemblyAliases(state, res, mat_vals);

  vec_handler.resizeOrZero(res, map.numRes());
  vec_handler.zero(jac.vals().view());

  const auto geom_v = geom.view();
  const auto map_v  = map.view();

  auto&             work     = detail::cpuWork();
  HostVector<Real>& state_e  = work.state;
  HostVector<Real>& coords_e = work.coords;
  HostVector<Real>& res_e    = work.res;
  HostVector<Real>& jac_e    = work.jac;
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

    const HostElementView elem{
        ie, geom.dim(), num_nodes, state_e.view(), coords_e.view()};

    for (Index row = 0; row < num_rows; ++row)
    {
      HostVectorView<Real> jac_row(jac_e.data() + row * num_cols, num_cols);
      kernel.evalRow(elem, row, res_e[row], jac_row);
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
 * @brief Assemble a time residual and state Jacobian over an element range.
 *
 * The element kernel implements
 * `evalRow(TimeElementView<Host>, wrt, row, res, jac_row)`. History storage is
 * lag-major with `map.numStates()` global values per lag.
 *
 * @tparam ElementKernel Row-wise time element evaluator.
 * @param[in] kernel - Element evaluator.
 * @param[in] step - Residual step index.
 * @param[in] num_hist - Number of history states.
 * @param[in] wrt - State block differentiated by the Jacobian.
 * @param[in] map - Element-to-global assembly map.
 * @param[in] element_begin - First element to assemble.
 * @param[in] element_end - One past the last element to assemble.
 * @param[in] hist - Global lag-major history states.
 * @param[in] nxt - Global next state.
 * @param[out] res - Global residual replaced by the assembled result.
 * @param[in,out] jac - Matrix zeroed and assembled in place.
 * @param[in] ctx - Execution context matching `jac`.
 * @throws std::runtime_error - If dimensions, the range, matrix layout, or
 * aliasing are invalid, or if the backend reports an error.
 */
template <class ElementKernel>
void assemble(const ElementKernel&                 kernel,
              Index                                step,
              Index                                num_hist,
              state::VariableBlock                 wrt,
              const HostAssemblyMap&               map,
              Index                                element_begin,
              Index                                element_end,
              HostVectorView<const Real>           hist,
              HostVectorView<const Real>           nxt,
              HostVector<Real>&                    res,
              linalg::Jacobian<MemorySpace::Host>& jac,
              linalg::Context<MemorySpace::Host>&  ctx)
{
  detail::checkTimeAssemblyInputs(num_hist, wrt, map, hist, nxt);
  detail::checkElementRange(map, element_begin, element_end);
  detail::checkTimeAssemblyAliases(hist, nxt, res);

  ctx.vectors().resizeOrZero(res, map.numRes());

  const auto map_v = map.view();
#pragma omp parallel
  {
    auto& work = detail::cpuWork();
    work.hist.reserve(num_hist * map.maxState());
    work.nxt.reserve(map.maxState());

#pragma omp for
    for (Index ie = element_begin; ie < element_end; ++ie)
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
      res, element_begin, element_end, map.numElems(), ctx);
}

/**
 * @brief Assemble a time residual over an element range.
 *
 * @tparam ElementKernel Row-wise time element evaluator.
 * @tparam Context CPU or PETSc execution context.
 * @param[in] kernel - Element evaluator.
 * @param[in] step - Residual step index.
 * @param[in] num_hist - Number of history states.
 * @param[in] map - Element-to-global assembly map.
 * @param[in] element_begin - First element to assemble.
 * @param[in] element_end - One past the last element to assemble.
 * @param[in] hist - Global lag-major history states.
 * @param[in] nxt - Global next state.
 * @param[out] res - Global residual replaced by the assembled result.
 * @param[in] ctx - Execution context.
 * @throws std::runtime_error - If dimensions, the range, or aliasing are
 * invalid, or if the backend reports an error.
 */
template <class ElementKernel, class Context>
void assembleResidual(
    const ElementKernel&       kernel,
    Index                      step,
    Index                      num_hist,
    const HostAssemblyMap&     map,
    Index                      element_begin,
    Index                      element_end,
    HostVectorView<const Real> hist,
    HostVectorView<const Real> nxt,
    HostVector<Real>&          res,
    Context&                   ctx)
{
  detail::checkTimeAssemblyInputs(
      num_hist, state::VariableBlock::NextState, map, hist, nxt);
  detail::checkElementRange(map, element_begin, element_end);
  detail::checkTimeAssemblyAliases(hist, nxt, res);
  ctx.vectors().resizeOrZero(res, map.numRes());

  const auto map_v = map.view();
#pragma omp parallel
  {
    auto& work = detail::cpuWork();
    work.hist.reserve(num_hist * map.maxState());
    work.nxt.reserve(map.maxState());

#pragma omp for
    for (Index ie = element_begin; ie < element_end; ++ie)
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
      res, element_begin, element_end, map.numElems(), ctx);
}

} // namespace assembly
} // namespace femx
