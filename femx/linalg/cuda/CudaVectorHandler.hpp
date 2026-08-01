#pragma once

#include <cstddef>

#include <femx/linalg/VectorHandler.hpp>

namespace femx::linalg
{

class CudaContext;

/**
 * @brief Implement Device vector operations with CUDA.
 *
 * Operations are enqueued on the stream owned by the bound CUDA context.
 */
class CudaVectorHandler final : public VectorHandler<MemorySpace::Device>
{
public:
  explicit CudaVectorHandler(CudaContext& ctx) noexcept;

  using VectorHandler<MemorySpace::Device>::copy;

  void assign(DeviceVector<Real>& out,
              Index               size,
              Real                val) const override;

  void assign(DeviceVector<Index>& out,
              Index                size,
              Index                val) const override;

  void copy(DeviceVectorView<const Real> src,
            DeviceVectorView<Real>       dst) const override;

  void copy(DeviceVectorView<const Real> src,
            DeviceVector<Real>&          dst) const override;

  void copy(HostVectorView<const Real> src,
            DeviceVectorView<Real>     dst) const override;

  void copy(HostVectorView<const Real> src,
            DeviceVector<Real>&        dst) const override;

  void copy(DeviceVectorView<const Real> src,
            HostVectorView<Real>         dst) const override;

  void copy(DeviceVectorView<const Real> src,
            HostVector<Real>&            dst) const override;

  void zero(DeviceVectorView<Real> vals) const override;

  void axpby(Real                         a,
             DeviceVectorView<const Real> x,
             Real                         b,
             DeviceVectorView<Real>       y) const override;

  void gather(DeviceVectorView<const Real>  src,
              DeviceVectorView<const Index> indices,
              DeviceVectorView<Real>        dst) const override;

  void scatter(DeviceVectorView<const Real>  src,
               DeviceVectorView<const Index> indices,
               DeviceVectorView<Real>        dst) const override;

  void dot(DeviceVectorView<const Real> x,
           DeviceVectorView<const Real> y,
           DeviceVectorView<Real>       out) const override;

private:
  void copyBytes(const void* src,
                 MemorySpace src_space,
                 void*       dst,
                 MemorySpace dst_space,
                 std::size_t bytes) const override;

  void* stream() const noexcept;

  CudaContext& ctx_; ///< Bound CUDA context.
};

} // namespace femx::linalg
