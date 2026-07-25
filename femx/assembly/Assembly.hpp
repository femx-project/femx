#pragma once

#include <algorithm>

#include <femx/assembly/AssemblyMap.hpp>
#include <femx/common/Checks.hpp>
#include <femx/common/Context.hpp>
#include <femx/fem/Geometry.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/handler/MatrixHandler.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>
#include <femx/state/TimeResidual.hpp>

#if defined(FEMX_HAS_PETSC)
#include <femx/linalg/petsc/PETScBackend.hpp>
#endif

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
void replaceRows(HostCsrMatrix&      mat,
                 const Array<Index>& rows,
                 Real                diag);

/** @brief Eliminate selected Host CSR columns and correct the right-hand side. */
void eliminateColumns(HostCsrMatrix&      mat,
                      const Array<Index>& rows,
                      HostVector&         rhs);

/// @cond INTERNAL
namespace detail
{
struct CpuWork
{
  HostVector   state;
  HostVector   coords;
  HostVector   hist;
  HostVector   nxt;
  HostVector   res;
  HostVector   jac;
  DenseMatrix  mat;
  Array<Index> rows;
  Array<Index> cols;
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
                                const HostVector&        state,
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
    Index                  num_hist,
    state::VariableBlock   wrt,
    const HostAssemblyMap& map,
    HostConstVectorView    hist,
    HostConstVectorView    nxt)
{
  require(num_hist > 0 && hist.size() == num_hist * map.numStates()
              && nxt.size() == map.numStates(),
          "Time assembly state dimensions do not match AssemblyMap");
  require(!wrt.isParam()
              && (!wrt.isHistoryState() || (wrt.historyLag() >= 0 && wrt.historyLag() < num_hist)),
          "Time assembly variable block is invalid");
}

inline void checkTimeAssemblyMatrix(
    const HostAssemblyMap& map,
    const HostCsrMatrix&   jac)
{
  require(jac.pattern().layoutId() == map.pattern().layoutId(),
          "Time assembly matrix must use the AssemblyMap CSR layout");
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

inline void checkTimeAssemblyAliases(HostConstVectorView hist,
                                     HostConstVectorView nxt,
                                     const HostVector&   res)
{
  require(!femx::detail::overlaps(hist, res.view())
              && !femx::detail::overlaps(nxt, res.view()),
          "Time assembly residual must not alias its inputs");
}

inline void checkTimeMatrixAlias(const HostVector&    res,
                                 const HostCsrMatrix& jac)
{
  require(&res != &jac.vals(),
          "Time assembly residual and matrix values must not alias");
}

inline void resizeOrZero(HostVector& out, Index size)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    std::fill(out.begin(), out.end(), Real{});
  }
}

inline void resetTimeMatrix(
    const HostAssemblyMap& map,
    HostCsrMatrix&         jac,
    CpuContext&            ctx)
{
  checkTimeAssemblyMatrix(map, jac);
  linalg::HostMatrixHandler mat_handler(ctx);
  mat_handler.zero(jac);
}

inline void addTimeElement(
    const HostAssemblyMap& map,
    Index                  ie,
    const DenseMatrix&     elem_mat,
    Array<Index>&,
    Array<Index>&,
    HostCsrMatrix& jac)
{
  addElem(map, ie, elem_mat, jac, true);
}

inline void reduceTimeResidual(HostVector&,
                               Index,
                               Index,
                               Index,
                               CpuContext&)
{
  // CPU assembly already accumulates into a single residual vector.
}

#if defined(FEMX_HAS_PETSC)
inline void checkTimeMatrixAlias(const HostVector&,
                                 const linalg::PETScOperator&)
{
}

inline void resetTimeMatrix(
    const HostAssemblyMap& map,
    linalg::PETScOperator& jac,
    linalg::PetscContext&  ctx)
{
  linalg::detail::checkInit();
  int       comm_relation = MPI_UNEQUAL;
  const int ierr =
      MPI_Comm_compare(jac.comm(), ctx.comm, &comm_relation);
  require(ierr == MPI_SUCCESS
              && (comm_relation == MPI_IDENT
                  || comm_relation == MPI_CONGRUENT),
          "PETSc time assembly matrix and context communicators must match");
  jac.resize(map.pattern());
}

inline void addTimeElement(
    const HostAssemblyMap& map,
    Index                  ie,
    const DenseMatrix&     elem_mat,
    Array<Index>&          rows,
    Array<Index>&          cols,
    linalg::PETScOperator& jac)
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
#pragma omp critical(femx_petsc_matrix_set_value)
  {
    jac.addBlock(rows, cols, elem_mat);
  }
}

inline void reduceTimeResidual(HostVector& res,
                               Index,
                               Index,
                               Index,
                               linalg::PetscContext& ctx)
{
  int comm_size = 0;
  int ierr      = MPI_Comm_size(ctx.comm, &comm_size);
  require(ierr == MPI_SUCCESS,
          "PETSc time assembly communicator query failed");
  if (comm_size == 1)
  {
    return;
  }
  ierr = MPI_Allreduce(MPI_IN_PLACE,
                       res.data(),
                       static_cast<int>(res.size()),
                       MPIU_REAL,
                       MPI_SUM,
                       ctx.comm);
  require(ierr == MPI_SUCCESS,
          "PETSc time assembly residual MPI reduction failed");
}
#endif

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
void assemble(const ElementKernel&     kernel,
              const fem::HostGeometry& geom,
              const HostAssemblyMap&   map,
              const HostVector&        state,
              HostVector&              res,
              HostCsrMatrix&           jac,
              CpuContext&              ctx)
{
  linalg::HostVectorHandler vec_handler(ctx);
  linalg::HostMatrixHandler mat_handler(ctx);
  detail::checkAssemblyInputs(geom, map, state, jac);
  const HostVector& mat_vals = jac.vals();
  detail::checkAssemblyAliases(state, res, mat_vals);

  vec_handler.resizeOrZero(res, map.numRes());
  mat_handler.zero(jac);

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

    const HostElementView elem{
        ie, geom.dim(), num_nodes, state_e.view(), coords_e.view()};

    for (Index row = 0; row < num_rows; ++row)
    {
      HostVectorView jac_row(jac_e.data() + row * num_cols, num_cols);
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
 * @tparam Matrix Host CSR or PETSc matrix.
 * @tparam Context CPU or PETSc execution context.
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
template <class ElementKernel, class Matrix, class Context>
void assemble(const ElementKernel&   kernel,
              Index                  step,
              Index                  num_hist,
              state::VariableBlock   wrt,
              const HostAssemblyMap& map,
              Index                  element_begin,
              Index                  element_end,
              HostConstVectorView    hist,
              HostConstVectorView    nxt,
              HostVector&            res,
              Matrix&                jac,
              Context&               ctx)
{
  detail::checkTimeAssemblyInputs(num_hist, wrt, map, hist, nxt);
  detail::checkElementRange(map, element_begin, element_end);
  detail::checkTimeAssemblyAliases(hist, nxt, res);
  detail::checkTimeMatrixAlias(res, jac);

  detail::resizeOrZero(res, map.numRes());
  detail::resetTimeMatrix(map, jac, ctx);

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
    const ElementKernel&   kernel,
    Index                  step,
    Index                  num_hist,
    const HostAssemblyMap& map,
    Index                  element_begin,
    Index                  element_end,
    HostConstVectorView    hist,
    HostConstVectorView    nxt,
    HostVector&            res,
    Context&               ctx)
{
  detail::checkTimeAssemblyInputs(
      num_hist, state::VariableBlock::NextState, map, hist, nxt);
  detail::checkElementRange(map, element_begin, element_end);
  detail::checkTimeAssemblyAliases(hist, nxt, res);
  detail::resizeOrZero(res, map.numRes());

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
                       HostVectorView{});
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
