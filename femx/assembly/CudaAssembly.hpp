#pragma once

#if defined(FEMX_HAS_CUDA) && defined(__CUDACC__)

#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <femx/assembly/Assembly.hpp>
#include <femx/common/Checks.hpp>

namespace femx
{
namespace assembly
{
/// @cond INTERNAL
namespace detail
{

inline void checkCudaStatus(cudaError_t status, const char* op)
{
  if (status != cudaSuccess)
  {
    throw std::runtime_error(std::string(op) + ": "
                             + cudaGetErrorString(status));
  }
}

inline void checkAssemblyInputs(
    const fem::DeviceGeometry&              geom,
    const AssemblyMap<MemorySpace::Device>& map,
    const DeviceVector&                     state,
    const DeviceCsrMatrix&                  jac)
{
  require(geom.numElems() == map.numElems(),
          "Geometry and AssemblyMap have different element counts");
  require(state.size() == map.numStates(),
          "Assembly state size does not match AssemblyMap");
  require(jac.graph().layoutId() == map.graph().layoutId(),
          "Assembly matrix must use the AssemblyMap CSR layout");
}

inline void checkTimeAssemblyInputs(
    Index                                   num_hist,
    state::VariableBlock                    wrt,
    const AssemblyMap<MemorySpace::Device>& map,
    DeviceConstVectorView                   hist,
    DeviceConstVectorView                   nxt)
{
  require(num_hist > 0 && hist.size() == num_hist * map.numStates()
              && nxt.size() == map.numStates(),
          "CUDA time assembly state dimensions do not match AssemblyMap");
  require(!wrt.isParam()
              && (!wrt.isHistoryState() || (wrt.historyLag() >= 0 && wrt.historyLag() < num_hist)),
          "CUDA time assembly variable block is invalid");
}

inline void checkTimeAssemblyInputs(
    Index                                   num_hist,
    state::VariableBlock                    wrt,
    const AssemblyMap<MemorySpace::Device>& map,
    DeviceConstVectorView                   hist,
    DeviceConstVectorView                   nxt,
    const DeviceCsrMatrix&                  jac)
{
  checkTimeAssemblyInputs(num_hist, wrt, map, hist, nxt);
  require(jac.graph().layoutId() == map.graph().layoutId(),
          "CUDA time assembly matrix must use the AssemblyMap CSR layout");
}

inline void checkTimeAssemblyAliases(DeviceConstVectorView hist,
                                     DeviceConstVectorView nxt,
                                     const DeviceVector&   res,
                                     const DeviceVector&   vals)
{
  require(hist.data() != res.data() && hist.data() != vals.data()
              && nxt.data() != res.data() && nxt.data() != vals.data()
              && res.data() != vals.data(),
          "CUDA time assembly outputs must not alias inputs or each other");
}

inline std::size_t assemblySharedBytes(
    const fem::DeviceGeometry&              geom,
    const AssemblyMap<MemorySpace::Device>& map)
{
  const auto count = static_cast<std::size_t>(map.maxState())
                     + static_cast<std::size_t>(geom.maxElemNodes())
                           * static_cast<std::size_t>(geom.dim())
                     + static_cast<std::size_t>(map.maxRes())
                     + static_cast<std::size_t>(map.maxJac());
  return count * sizeof(Real);
}

inline std::size_t timeAssemblySharedBytes(
    Index                                   num_hist,
    const AssemblyMap<MemorySpace::Device>& map)
{
  const auto count =
      static_cast<std::size_t>(num_hist + 1)
          * static_cast<std::size_t>(map.maxState())
      + static_cast<std::size_t>(map.maxRes())
      + static_cast<std::size_t>(map.maxJac());
  return count * sizeof(Real);
}

template <class ElementOperator>
__global__ void assembleKernel(
    ElementOperator                        op,
    fem::GeometryView<MemorySpace::Device> geom,
    AssemblyMapView<MemorySpace::Device>   map,
    const Real*                            state,
    Real*                                  res,
    Real*                                  jac)
{
  const Index ie = static_cast<Index>(blockIdx.x);
  if (ie >= map.num_elems)
  {
    return;
  }

  const Index num_rows   = map.numResDofs(ie);
  const Index num_cols   = map.numStateDofs(ie);
  const Index num_nodes  = geom.elemNumNodes(ie);
  const Index num_coords = num_nodes * geom.dim();
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
    const Index in   = i / geom.dim();
    const Index d    = i - in * geom.dim();
    const Index node = geom.elemNode(ie, in);
    coords_e[i]      = geom.coord(node, d);
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

  const ElementView<MemorySpace::Device> elem{
      ie, geom.dim(), num_nodes, {state_e, num_cols}, {coords_e, num_coords}};

  for (Index row = tid; row < num_rows; row += stride)
  {
    VectorView<MemorySpace::Device, Real> jac_row(jac_e + row * num_cols,
                                                  num_cols);
    op.evalRow(elem, row, res_e[row], jac_row);
  }
  __syncthreads();

  for (Index row = tid; row < num_rows; row += stride)
  {
    atomicAdd(res + map.resDof(ie, row), res_e[row]);
  }
  for (Index i = tid; i < num_jac; i += stride)
  {
    atomicAdd(jac + map.jacIndex(ie, i), jac_e[i]);
  }
}

template <class ElementOperator>
__global__ void assembleTimeKernel(
    ElementOperator                      op,
    Index                                step,
    Index                                num_hist,
    state::VariableBlock                 wrt,
    AssemblyMapView<MemorySpace::Device> map,
    const Real*                          hist,
    const Real*                          nxt,
    Real*                                res,
    Real*                                jac)
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

  const TimeElementView<MemorySpace::Device> elem{
      ie,
      step,
      num_hist,
      {hist_e, num_hist * ncol},
      {nxt_e, ncol}};
  for (Index row = tid; row < nrow; row += stride)
  {
    VectorView<MemorySpace::Device, Real> jac_row(jac_e + row * ncol, ncol);
    op.evalRow(elem, wrt, row, res_e[row], jac_row);
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

template <class ElementOperator>
int configureAssemblyLaunch(std::size_t smem)
{
  constexpr int threads = 128;
  int           dev     = 0;
  checkCudaStatus(cudaGetDevice(&dev), "cudaGetDevice failed for CUDA assembly");

  int default_smem = 0;
  checkCudaStatus(
      cudaDeviceGetAttribute(
          &default_smem, cudaDevAttrMaxSharedMemoryPerBlock, dev),
      "cudaDeviceGetAttribute(shared memory) failed for CUDA assembly");
  if (smem > static_cast<std::size_t>(default_smem))
  {
    checkCudaStatus(
        cudaFuncSetAttribute(
            assembleKernel<ElementOperator>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)),
        "cudaFuncSetAttribute failed for CUDA assembly");
  }

  return threads;
}

template <class ElementOperator>
int configureTimeAssemblyLaunch(std::size_t smem)
{
  constexpr int threads = 128;
  int           dev     = 0;
  checkCudaStatus(cudaGetDevice(&dev), "cudaGetDevice failed for CUDA time assembly");

  int default_smem = 0;
  checkCudaStatus(
      cudaDeviceGetAttribute(
          &default_smem, cudaDevAttrMaxSharedMemoryPerBlock, dev),
      "cudaDeviceGetAttribute(shared memory) failed for CUDA time assembly");
  if (smem > static_cast<std::size_t>(default_smem))
  {
    checkCudaStatus(
        cudaFuncSetAttribute(
            assembleTimeKernel<ElementOperator>,
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
 * ElementOperator is the only template parameter. It implements the same
 * row-wise `evalRow` contract as the CPU overload, with device ElementView and
 * VectorView arguments. All zeroing and the assembly launch are enqueued on
 * the CudaContext stream; callers synchronize only at an explicit boundary.
 *
 * @tparam ElementOperator Trivially copyable row-wise element evaluator.
 * @param op Element evaluator copied into the kernel launch.
 * @param geom Device geometry matching the map's element order.
 * @param map Device element-to-global assembly map.
 * @param state Global device state vector.
 * @param res Device residual replaced by the assembled result.
 * @param jac Device CSR matrix zeroed and assembled in place.
 * @param ctx CUDA stream on which all work is enqueued.
 */
template <class ElementOperator>
void assemble(const ElementOperator&                  op,
              const fem::DeviceGeometry&              geom,
              const AssemblyMap<MemorySpace::Device>& map,
              const DeviceVector&                     state,
              DeviceVector&                           res,
              DeviceCsrMatrix&                        jac,
              CudaContext&                            ctx)
{
  static_assert(std::is_trivially_copyable<ElementOperator>::value,
                "CUDA ElementOperator must be trivially copyable");

  detail::checkAssemblyInputs(geom, map, state, jac);
  const DeviceVector& mat_vals = jac.vals();
  detail::checkAssemblyAliases(state, res, mat_vals);

  if (res.size() != map.numRes())
  {
    res.resize(map.numRes());
  }
  res.setZero(ctx);
  jac.setZero(ctx);

  if (map.numElems() == 0)
  {
    return;
  }

  const std::size_t smem    = detail::assemblySharedBytes(geom, map);
  const int         threads = detail::configureAssemblyLaunch<ElementOperator>(smem);
  const auto        stream  = static_cast<cudaStream_t>(ctx.stream());

  detail::assembleKernel<ElementOperator>
      <<<static_cast<unsigned int>(map.numElems()),
         static_cast<unsigned int>(threads),
         smem,
         stream>>>(op,
                   geom.view(),
                   map.view(),
                   state.data(),
                   res.data(),
                   jac.valsData());
  device::checkLastError();
}

/** @brief Assemble one time residual and state Jacobian on CUDA. */
template <class ElementOperator>
void assemble(const ElementOperator&                  op,
              Index                                   step,
              Index                                   num_hist,
              state::VariableBlock                    wrt,
              const AssemblyMap<MemorySpace::Device>& map,
              DeviceConstVectorView                   hist,
              DeviceConstVectorView                   nxt,
              DeviceVector&                           res,
              DeviceCsrMatrix&                        jac,
              CudaContext&                            ctx)
{
  static_assert(std::is_trivially_copyable<ElementOperator>::value,
                "CUDA time ElementOperator must be trivially copyable");

  detail::checkTimeAssemblyInputs(num_hist, wrt, map, hist, nxt, jac);
  const DeviceVector& vals = jac.vals();
  detail::checkTimeAssemblyAliases(hist, nxt, res, vals);

  if (res.size() != map.numRes())
  {
    res.resize(map.numRes());
  }
  res.setZero(ctx);
  jac.setZero(ctx);
  if (map.numElems() == 0)
  {
    return;
  }

  const std::size_t smem    = detail::timeAssemblySharedBytes(num_hist, map);
  const int         threads = detail::configureTimeAssemblyLaunch<ElementOperator>(smem);
  const auto        stream  = static_cast<cudaStream_t>(ctx.stream());

  detail::assembleTimeKernel<ElementOperator>
      <<<static_cast<unsigned int>(map.numElems()),
         static_cast<unsigned int>(threads),
         smem,
         stream>>>(op,
                   step,
                   num_hist,
                   wrt,
                   map.view(),
                   hist.data(),
                   nxt.data(),
                   res.data(),
                   jac.valsData());
  device::checkLastError();
}

/** @brief Assemble one time residual on CUDA without allocating a Jacobian. */
template <class ElementOperator>
void assembleResidual(
    const ElementOperator&                  op,
    Index                                   step,
    Index                                   num_hist,
    const AssemblyMap<MemorySpace::Device>& map,
    DeviceConstVectorView                   hist,
    DeviceConstVectorView                   nxt,
    DeviceVector&                           res,
    CudaContext&                            ctx)
{
  static_assert(std::is_trivially_copyable<ElementOperator>::value,
                "CUDA time ElementOperator must be trivially copyable");

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
  res.setZero(ctx);
  if (map.numElems() == 0)
  {
    return;
  }

  const std::size_t smem = detail::timeAssemblySharedBytes(num_hist, map);
  const int         threads =
      detail::configureTimeAssemblyLaunch<ElementOperator>(smem);
  const auto stream = static_cast<cudaStream_t>(ctx.stream());

  detail::assembleTimeKernel<ElementOperator>
      <<<static_cast<unsigned int>(map.numElems()),
         static_cast<unsigned int>(threads),
         smem,
         stream>>>(op,
                   step,
                   num_hist,
                   state::VariableBlock::NextState,
                   map.view(),
                   hist.data(),
                   nxt.data(),
                   res.data(),
                   nullptr);
  device::checkLastError();
}

} // namespace assembly
} // namespace femx

#endif
