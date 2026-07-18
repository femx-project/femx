#pragma once

#if defined(FEMX_HAS_CUDA) && defined(__CUDACC__)

#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <femx/assembly/Assembly.hpp>

namespace femx
{
namespace assembly
{
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

inline std::size_t assemblySharedBytes(
    const fem::DeviceGeometry&              geom,
    const AssemblyMap<MemorySpace::Device>& map)
{
  const auto count = static_cast<std::size_t>(map.maxStateDofs())
                     + static_cast<std::size_t>(geom.maxElemNodes())
                           * static_cast<std::size_t>(geom.dim())
                     + static_cast<std::size_t>(map.maxResDofs())
                     + static_cast<std::size_t>(map.maxJacEntries());
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
int configureAssemblyLaunch(std::size_t smem)
{
  constexpr int threads = 128;
  int           dev     = 0;
  checkCudaStatus(cudaGetDevice(&dev),
                  "cudaGetDevice failed for CUDA assembly");

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

} // namespace detail

/**
 * @brief Assemble residual and Jacobian with one CUDA block per element.
 *
 * ElementOperator is the only template parameter. It implements the same
 * row-wise `evalRow` contract as the CPU overload, with device ElementView and
 * VectorView arguments. All zeroing and the assembly launch are enqueued on
 * the CudaContext stream; callers synchronize only at an explicit boundary.
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

  const std::size_t smem = detail::assemblySharedBytes(geom, map);
  const int         threads =
      detail::configureAssemblyLaunch<ElementOperator>(smem);
  const auto stream = static_cast<cudaStream_t>(ctx.stream());

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

} // namespace assembly
} // namespace femx

#endif
