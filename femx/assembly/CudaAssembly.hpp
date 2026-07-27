#pragma once

#if defined(FEMX_HAS_CUDA) && defined(__CUDACC__)

#include <cuda_runtime_api.h>

#include <cstddef>
#include <type_traits>

#include <femx/assembly/Assembly.hpp>
#include <femx/common/Checks.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaSystemMatrix.hpp>

namespace femx
{
namespace assembly
{
/// @cond INTERNAL
namespace detail
{

inline void checkAssemblyInputs(
    const fem::DeviceMesh&    mesh,
    const DeviceAssemblyMap&  map,
    const DeviceVector<Real>& state)
{
  require(mesh.numElems() == map.numElems(),
          "Mesh and AssemblyMap have different element counts");
  require(state.size() == map.numStates(),
          "Assembly state size does not match AssemblyMap");
}

inline void checkAssemblyInputs(
    const fem::DeviceMesh&               mesh,
    const DeviceAssemblyMap&             map,
    const DeviceVector<Real>&            state,
    const linalg::DeviceCsrAssemblyView& jac)
{
  checkAssemblyInputs(mesh, map, state);
  require(jac.rows == map.pattern().rows()
              && jac.columns == map.pattern().cols()
              && jac.nonzeros == map.pattern().nnz(),
          "Assembly matrix dimensions do not match the AssemblyMap");
}

inline void checkTimeAssemblyInputs(
    Index                        num_hist,
    state::VariableBlock         wrt,
    const DeviceAssemblyMap&     map,
    DeviceVectorView<const Real> hist,
    DeviceVectorView<const Real> nxt)
{
  require(num_hist > 0 && hist.size() == num_hist * map.numStates()
              && nxt.size() == map.numStates(),
          "CUDA time assembly state dimensions do not match AssemblyMap");
  require(!wrt.isParam()
              && (!wrt.isHistoryState() || (wrt.historyLag() >= 0 && wrt.historyLag() < num_hist)),
          "CUDA time assembly variable block is invalid");
}

inline void checkTimeAssemblyInputs(
    Index                                num_hist,
    state::VariableBlock                 wrt,
    const DeviceAssemblyMap&             map,
    DeviceVectorView<const Real>         hist,
    DeviceVectorView<const Real>         nxt,
    const linalg::DeviceCsrAssemblyView& jac)
{
  checkTimeAssemblyInputs(num_hist, wrt, map, hist, nxt);
  require(jac.rows == map.pattern().rows()
              && jac.columns == map.pattern().cols()
              && jac.nonzeros == map.pattern().nnz(),
          "CUDA time assembly matrix dimensions do not match the AssemblyMap");
}

inline void checkTimeAssemblyAliases(DeviceVectorView<const Real> hist,
                                     DeviceVectorView<const Real> nxt,
                                     const DeviceVector<Real>&    res,
                                     const DeviceVector<Real>&    vals)
{
  require(hist.data() != res.data() && hist.data() != vals.data()
              && nxt.data() != res.data() && nxt.data() != vals.data()
              && res.data() != vals.data(),
          "CUDA time assembly outputs must not alias inputs or each other");
}

inline std::size_t assemblySharedBytes(
    const fem::DeviceMesh&   mesh,
    const DeviceAssemblyMap& map)
{
  const auto count = static_cast<std::size_t>(map.maxState())
                     + static_cast<std::size_t>(mesh.maxElemNodes())
                           * static_cast<std::size_t>(mesh.dim())
                     + static_cast<std::size_t>(map.maxRes())
                     + static_cast<std::size_t>(map.maxJac());
  return count * sizeof(Real);
}

inline std::size_t timeAssemblySharedBytes(
    Index                    num_hist,
    const DeviceAssemblyMap& map)
{
  const auto count =
      static_cast<std::size_t>(num_hist + 1)
          * static_cast<std::size_t>(map.maxState())
      + static_cast<std::size_t>(map.maxRes())
      + static_cast<std::size_t>(map.maxJac());
  return count * sizeof(Real);
}

template <class ElementKernel>
__global__ void assembleKernel(
    ElementKernel                      kernel,
    fem::MeshView<MemorySpace::Device> mesh,
    DeviceAssemblyMapView              map,
    const Real*                        state,
    Real*                              res,
    Real*                              jac)
{
  const Index ie = static_cast<Index>(blockIdx.x);
  if (ie >= map.num_elems)
  {
    return;
  }

  const Index num_rows   = map.numResDofs(ie);
  const Index num_cols   = map.numStateDofs(ie);
  const Index num_nodes  = mesh.elemNumNodes(ie);
  const Index num_coords = num_nodes * mesh.dim();
  const Index num_jac    = num_rows * num_cols;
  const Index tid        = static_cast<Index>(threadIdx.x);
  const Index stride     = static_cast<Index>(blockDim.x);

  extern __shared__ Real work[];
  Real*                  state_e  = work;
  Real*                  coords_e = state_e + num_cols;
  Real*                  res_e    = coords_e + num_coords;
  Real*                  jac_e    = res_e + num_rows;

  for (Index col = tid; col < num_cols; col += stride)
  {
    state_e[col] = state[map.stateDof(ie, col)];
  }
  for (Index i = tid; i < num_coords; i += stride)
  {
    const Index in   = i / mesh.dim();
    const Index d    = i - in * mesh.dim();
    const Index node = mesh.elemNode(ie, in);
    coords_e[i]      = mesh.coord(node, d);
  }
  for (Index row = tid; row < num_rows; row += stride)
  {
    res_e[row] = Real{};
  }
  for (Index i = tid; i < num_jac; i += stride)
  {
    jac_e[i] = Real{};
  }
  __syncthreads();

  const DeviceElementView elem{
      ie, mesh.dim(), num_nodes, {state_e, num_cols}, {coords_e, num_coords}};

  for (Index row = tid; row < num_rows; row += stride)
  {
    VectorView<MemorySpace::Device, Real> jac_row(jac_e + row * num_cols,
                                                  num_cols);
    kernel.evalRow(elem, row, res_e[row], jac_row);
  }
  __syncthreads();

  if (res != nullptr)
  {
    for (Index row = tid; row < num_rows; row += stride)
    {
      atomicAdd(res + map.resDof(ie, row), res_e[row]);
    }
  }
  if (jac != nullptr)
  {
    for (Index i = tid; i < num_jac; i += stride)
    {
      atomicAdd(jac + map.jacIndex(ie, i), jac_e[i]);
    }
  }
}

template <class ElementKernel>
__global__ void assembleTimeKernel(
    ElementKernel         kernel,
    Index                 step,
    Index                 num_hist,
    state::VariableBlock  wrt,
    DeviceAssemblyMapView map,
    const Real*           hist,
    const Real*           nxt,
    Real*                 res,
    Real*                 jac)
{
  const Index ie = static_cast<Index>(blockIdx.x);
  if (ie >= map.num_elems)
  {
    return;
  }

  const Index nrow   = map.numResDofs(ie);
  const Index ncol   = map.numStateDofs(ie);
  const Index njac   = nrow * ncol;
  const Index tid    = static_cast<Index>(threadIdx.x);
  const Index stride = static_cast<Index>(blockDim.x);

  extern __shared__ Real work[];
  Real*                  hist_e = work;
  Real*                  nxt_e  = hist_e + num_hist * ncol;
  Real*                  res_e  = nxt_e + ncol;
  Real*                  jac_e  = res_e + nrow;

  for (Index i = tid; i < num_hist * ncol; i += stride)
  {
    const Index lag = i / ncol;
    const Index col = i - lag * ncol;
    const Index dof = map.stateDof(ie, col);
    hist_e[i]       = hist[lag * map.num_states + dof];
  }
  for (Index col = tid; col < ncol; col += stride)
  {
    nxt_e[col] = nxt[map.stateDof(ie, col)];
  }
  for (Index row = tid; row < nrow; row += stride)
  {
    res_e[row] = Real{};
  }
  for (Index i = tid; i < njac; i += stride)
  {
    jac_e[i] = Real{};
  }
  __syncthreads();

  const DeviceTimeElementView elem{
      ie,
      step,
      num_hist,
      {hist_e, num_hist * ncol},
      {nxt_e, ncol}};
  for (Index row = tid; row < nrow; row += stride)
  {
    VectorView<MemorySpace::Device, Real> jac_row(jac_e + row * ncol, ncol);
    kernel.evalRow(elem, wrt, row, res_e[row], jac_row);
  }
  __syncthreads();

  for (Index row = tid; row < nrow; row += stride)
  {
    atomicAdd(res + map.resDof(ie, row), res_e[row]);
  }
  if (jac != nullptr)
  {
    for (Index i = tid; i < njac; i += stride)
    {
      atomicAdd(jac + map.jacIndex(ie, i), jac_e[i]);
    }
  }
}

template <class ElementKernel>
int checkAssemblyLaunch(std::size_t smem)
{
  constexpr int threads = 128;
  int           dev     = 0;
  cuda::check(cudaGetDevice(&dev), "cudaGetDevice failed for CUDA assembly");

  int default_smem = 0;
  cuda::check(
      cudaDeviceGetAttribute(
          &default_smem, cudaDevAttrMaxSharedMemoryPerBlock, dev),
      "cudaDeviceGetAttribute(shared memory) failed for CUDA assembly");
  if (smem > static_cast<std::size_t>(default_smem))
  {
    cuda::check(
        cudaFuncSetAttribute(
            assembleKernel<ElementKernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)),
        "cudaFuncSetAttribute failed for CUDA assembly");
  }

  return threads;
}

template <class ElementKernel>
int checkTimeAssemblyLaunch(std::size_t smem)
{
  constexpr int threads = 128;
  int           dev     = 0;
  cuda::check(cudaGetDevice(&dev),
              "cudaGetDevice failed for CUDA time assembly");

  int default_smem = 0;
  cuda::check(
      cudaDeviceGetAttribute(
          &default_smem, cudaDevAttrMaxSharedMemoryPerBlock, dev),
      "cudaDeviceGetAttribute(shared memory) failed for CUDA time assembly");
  if (smem > static_cast<std::size_t>(default_smem))
  {
    cuda::check(
        cudaFuncSetAttribute(
            assembleTimeKernel<ElementKernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)),
        "cudaFuncSetAttribute failed for CUDA time assembly");
  }
  return threads;
}

} // namespace detail

/// @endcond

/**
 * @brief Assemble residual and Jacobian with one CUDA block per element.
 *
 * @param[in] kernel - Element evaluator copied into the kernel launch.
 * @param[in] mesh - Device mesh matching the map's element order.
 * @param[in] map - Device element-to-global assembly map.
 * @param[in] state - Global Device state vector.
 * @param[out] res - Device residual replaced by the assembled result.
 * @param[in,out] jac - Device CSR matrix zeroed and assembled in place.
 * @param[in,out] ctx - CUDA context on which all work is enqueued.
 */
template <class ElementKernel>
void assembleResidualAndJacobian(
    const ElementKernel&      kernel,
    const fem::DeviceMesh&    mesh,
    const DeviceAssemblyMap&  map,
    const DeviceVector<Real>& state,
    DeviceVector<Real>&       res,
    linalg::CudaSystemMatrix& jac,
    linalg::CudaContext&      ctx)
{
  auto& vec_handler = ctx.vectors();
  static_assert(std::is_trivially_copyable<ElementKernel>::value,
                "CUDA element kernel must be trivially copyable");

  const auto jac_view = jac.assemblyView();
  detail::checkAssemblyInputs(mesh, map, state, jac_view);
  require(state.data() != res.data()
              && state.data() != jac_view.values.data()
              && res.data() != jac_view.values.data(),
          "Assembly state, residual, and matrix values must not alias");

  if (res.size() != map.numRes())
  {
    res.resize(map.numRes());
  }
  vec_handler.zero(res.view());

  if (map.numElems() == 0)
  {
    return;
  }

  const std::size_t smem    = detail::assemblySharedBytes(mesh, map);
  const int         threads = detail::checkAssemblyLaunch<ElementKernel>(smem);
  const auto        stream  = static_cast<cudaStream_t>(ctx.stream());

  detail::assembleKernel<ElementKernel>
      <<<static_cast<unsigned int>(map.numElems()),
         static_cast<unsigned int>(threads),
         smem,
         stream>>>(kernel,
                   mesh.view(),
                   map.view(),
                   state.data(),
                   res.data(),
                   jac_view.values.data());
  cuda::checkLastError();
}

/**
 * @brief Assemble a stationary Device residual without a Jacobian.
 *
 * @param[in] kernel - Element evaluator copied into the kernel launch.
 * @param[in] mesh - Device mesh matching the map's element order.
 * @param[in] map - Device element-to-global assembly map.
 * @param[in] state - Global Device state vector.
 * @param[out] res - Device residual replaced by the assembled result.
 * @param[in,out] ctx - CUDA context on which all work is enqueued.
 */
template <class ElementKernel>
void assembleResidual(const ElementKernel&      kernel,
                      const fem::DeviceMesh&    mesh,
                      const DeviceAssemblyMap&  map,
                      const DeviceVector<Real>& state,
                      DeviceVector<Real>&       res,
                      linalg::CudaContext&      ctx)
{
  auto& vec_handler = ctx.vectors();
  static_assert(std::is_trivially_copyable<ElementKernel>::value,
                "CUDA element kernel must be trivially copyable");

  detail::checkAssemblyInputs(mesh, map, state);
  require(state.data() != res.data(),
          "Assembly state and residual must not alias");

  if (res.size() != map.numRes())
  {
    res.resize(map.numRes());
  }
  vec_handler.zero(res.view());
  if (map.numElems() == 0)
  {
    return;
  }

  const std::size_t smem    = detail::assemblySharedBytes(mesh, map);
  const int         threads = detail::checkAssemblyLaunch<ElementKernel>(smem);
  const auto        stream  = static_cast<cudaStream_t>(ctx.stream());

  detail::assembleKernel<ElementKernel>
      <<<static_cast<unsigned int>(map.numElems()),
         static_cast<unsigned int>(threads),
         smem,
         stream>>>(kernel,
                   mesh.view(),
                   map.view(),
                   state.data(),
                   res.data(),
                   nullptr);
  cuda::checkLastError();
}

/**
 * @brief Assemble a stationary Device Jacobian without a residual.
 *
 * @param[in] kernel - Element evaluator copied into the kernel launch.
 * @param[in] mesh - Device mesh matching the map's element order.
 * @param[in] map - Device element-to-global assembly map.
 * @param[in] state - Global Device state vector.
 * @param[in,out] jacobian - Device CSR Jacobian receiving element contributions.
 * @param[in,out] ctx - CUDA context on which all work is enqueued.
 */
template <class ElementKernel>
void assembleJacobian(
    const ElementKernel&      kernel,
    const fem::DeviceMesh&    mesh,
    const DeviceAssemblyMap&  map,
    const DeviceVector<Real>& state,
    linalg::CudaSystemMatrix& jacobian,
    linalg::CudaContext&      ctx)
{
  static_assert(std::is_trivially_copyable<ElementKernel>::value,
                "CUDA element kernel must be trivially copyable");

  const auto jac_view = jacobian.assemblyView();
  detail::checkAssemblyInputs(mesh, map, state, jac_view);
  require(state.data() != jac_view.values.data(),
          "Assembly state and matrix values must not alias");

  if (map.numElems() == 0)
  {
    return;
  }

  const std::size_t shared_bytes = detail::assemblySharedBytes(mesh, map);
  const int         threads      = detail::checkAssemblyLaunch<ElementKernel>(shared_bytes);
  const auto        stream       = static_cast<cudaStream_t>(ctx.stream());

  detail::assembleKernel<ElementKernel>
      <<<static_cast<unsigned int>(map.numElems()),
         static_cast<unsigned int>(threads),
         shared_bytes,
         stream>>>(kernel,
                   mesh.view(),
                   map.view(),
                   state.data(),
                   nullptr,
                   jac_view.values.data());
  cuda::checkLastError();
}

/**
 * @brief Assemble one time residual and state Jacobian on CUDA.
 *
 * @param[in] kernel - Element evaluator copied into the kernel launch.
 * @param[in] step - Residual step index.
 * @param[in] num_hist - Number of history states.
 * @param[in] wrt - State block differentiated by the Jacobian.
 * @param[in] map - Device element-to-global assembly map.
 * @param[in] hist - Global Device history states.
 * @param[in] nxt - Global Device next state.
 * @param[out] res - Device residual replaced by the assembled result.
 * @param[in,out] jac - Device CSR matrix zeroed and assembled in place.
 * @param[in,out] ctx - CUDA context on which all work is enqueued.
 */
template <class ElementKernel>
void assembleResidualAndJacobian(
    const ElementKernel&         kernel,
    Index                        step,
    Index                        num_hist,
    state::VariableBlock         wrt,
    const DeviceAssemblyMap&     map,
    DeviceVectorView<const Real> hist,
    DeviceVectorView<const Real> nxt,
    DeviceVector<Real>&          res,
    linalg::CudaSystemMatrix&    jac,
    linalg::CudaContext&         ctx)
{
  auto& vec_handler = ctx.vectors();
  static_assert(std::is_trivially_copyable<ElementKernel>::value,
                "CUDA time element kernel must be trivially copyable");

  const auto jac_view = jac.assemblyView();
  detail::checkTimeAssemblyInputs(
      num_hist, wrt, map, hist, nxt, jac_view);
  require(hist.data() != res.data()
              && hist.data() != jac_view.values.data()
              && nxt.data() != res.data()
              && nxt.data() != jac_view.values.data()
              && res.data() != jac_view.values.data(),
          "CUDA time assembly outputs must not alias inputs or each other");

  if (res.size() != map.numRes())
  {
    res.resize(map.numRes());
  }
  vec_handler.zero(res.view());
  if (map.numElems() == 0)
  {
    return;
  }

  const std::size_t smem    = detail::timeAssemblySharedBytes(num_hist, map);
  const int         threads = detail::checkTimeAssemblyLaunch<ElementKernel>(smem);
  const auto        stream  = static_cast<cudaStream_t>(ctx.stream());

  detail::assembleTimeKernel<ElementKernel>
      <<<static_cast<unsigned int>(map.numElems()),
         static_cast<unsigned int>(threads),
         smem,
         stream>>>(kernel,
                   step,
                   num_hist,
                   wrt,
                   map.view(),
                   hist.data(),
                   nxt.data(),
                   res.data(),
                   jac_view.values.data());
  cuda::checkLastError();
}

/**
 * @brief Assemble one time residual on CUDA without allocating a Jacobian.
 *
 * @param[in] kernel - Element evaluator copied into the kernel launch.
 * @param[in] step - Residual step index.
 * @param[in] num_hist - Number of history states.
 * @param[in] map - Device element-to-global assembly map.
 * @param[in] hist - Global Device history states.
 * @param[in] nxt - Global Device next state.
 * @param[out] res - Device residual replaced by the assembled result.
 * @param[in,out] ctx - CUDA context on which all work is enqueued.
 */
template <class ElementKernel>
void assembleResidual(
    const ElementKernel&         kernel,
    Index                        step,
    Index                        num_hist,
    const DeviceAssemblyMap&     map,
    DeviceVectorView<const Real> hist,
    DeviceVectorView<const Real> nxt,
    DeviceVector<Real>&          res,
    linalg::CudaContext&         ctx)
{
  auto& vec_handler = ctx.vectors();
  static_assert(std::is_trivially_copyable<ElementKernel>::value,
                "CUDA time element kernel must be trivially copyable");

  detail::checkTimeAssemblyInputs(num_hist,
                                  state::VariableBlock::NextState,
                                  map,
                                  hist,
                                  nxt);
  require(hist.data() != res.data() && nxt.data() != res.data(),
          "CUDA time residual output must not alias its inputs");

  if (res.size() != map.numRes())
  {
    res.resize(map.numRes());
  }
  vec_handler.zero(res.view());
  if (map.numElems() == 0)
  {
    return;
  }

  const std::size_t smem    = detail::timeAssemblySharedBytes(num_hist, map);
  const int         threads = detail::checkTimeAssemblyLaunch<ElementKernel>(smem);
  const auto        stream  = static_cast<cudaStream_t>(ctx.stream());

  detail::assembleTimeKernel<ElementKernel>
      <<<static_cast<unsigned int>(map.numElems()),
         static_cast<unsigned int>(threads),
         smem,
         stream>>>(kernel,
                   step,
                   num_hist,
                   state::VariableBlock::NextState,
                   map.view(),
                   hist.data(),
                   nxt.data(),
                   res.data(),
                   nullptr);
  cuda::checkLastError();
}

} // namespace assembly
} // namespace femx

#endif
