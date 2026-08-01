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
  using Base = VectorHandler<MemorySpace::Device>;

public:
  explicit CudaVectorHandler(CudaContext& ctx) noexcept;

  using Base::copy;

  /**
   * @copydoc Base::assign(DeviceVector<Real>&,Index,Real) const
   */
  void assign(DeviceVector<Real>& out,
              Index               size,
              Real                val) const override;

  /**
   * @copydoc Base::assign(DeviceVector<Index>&,Index,Index) const
   */
  void assign(DeviceVector<Index>& out,
              Index                size,
              Index                val) const override;

  /**
   * @brief Copy between same-sized Device views.
   *
   * @param[in]  src - Device source view.
   * @param[out] dst - Device destination view.
   * @throws std::runtime_error If validation fails.
   */
  void copy(DeviceVectorView<const Real> src,
            DeviceVectorView<Real>       dst) const override;

  /**
   * @brief Resize an owning Device vector and copy a Device view into it.
   *
   * @param[in]  src - Device source view.
   * @param[out] dst - Owning Device destination resized to `src.size()`.
   * @throws std::runtime_error If validation fails.
   */
  void copy(DeviceVectorView<const Real> src,
            DeviceVector<Real>&          dst) const override;

  /**
   * @brief Copy a Host view into a same-sized Device view.
   *
   * @param[in]  src - Host source view.
   * @param[out] dst - Device destination view.
   * @throws std::runtime_error If validation fails.
   */
  void copy(HostVectorView<const Real> src,
            DeviceVectorView<Real>     dst) const override;

  /**
   * @brief Resize an owning Device vector and copy a Host view into it.
   *
   * @param[in]  src - Host source view.
   * @param[out] dst - Owning Device destination resized to `src.size()`.
   * @throws std::runtime_error If validation fails.
   */
  void copy(HostVectorView<const Real> src,
            DeviceVector<Real>&        dst) const override;

  /**
   * @brief Copy a Device view into a same-sized Host view.
   *
   * @param[in]  src - Device source view.
   * @param[out] dst - Host destination view.
   * @throws std::runtime_error If validation fails.
   */
  void copy(DeviceVectorView<const Real> src,
            HostVectorView<Real>         dst) const override;

  /**
   * @brief Resize an owning Host vector and copy a Device view into it.
   *
   * @param[in]  src - Device source view.
   * @param[out] dst - Owning Host destination resized to `src.size()`.
   * @throws std::runtime_error If validation fails.
   */
  void copy(DeviceVectorView<const Real> src,
            HostVector<Real>&            dst) const override;

  /**
   * @copydoc Base::zero()
   */
  void zero(DeviceVectorView<Real> vals) const override;

  /**
   * @copydoc Base::axpby()
   */
  void axpby(Real                         a,
             DeviceVectorView<const Real> x,
             Real                         b,
             DeviceVectorView<Real>       y) const override;

  /**
   * @copydoc Base::gather()
   */
  void gather(DeviceVectorView<const Real>  src,
              DeviceVectorView<const Index> idx,
              DeviceVectorView<Real>        dst) const override;

  /**
   * @copydoc Base::scatter()
   */
  void scatter(DeviceVectorView<const Real>  src,
               DeviceVectorView<const Index> idx,
               DeviceVectorView<Real>        dst) const override;

  /**
   * @copydoc Base::dot()
   */
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
