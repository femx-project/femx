#pragma once

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>
#include <femx/linalg/Context.hpp>

namespace femx
{
namespace fem
{

class FESpace;
class GaussQuadrature;

template <MemorySpace Space>
class ElementQuadData;

using HostElementQuadData   = ElementQuadData<MemorySpace::Host>;
using DeviceElementQuadData = ElementQuadData<MemorySpace::Device>;

/**
 * @brief Compute reusable element quadrature data on the Host.
 *
 * @param[in] space - Initialized finite-element space.
 * @param[in] quad  - Quadrature rule for the space's reference element.
 * @return Shape values, physical gradients, and weighted Jacobians for every
 * element.
 * @throws std::runtime_error If validation fails.
 */
HostElementQuadData makeElementQuadData(
    const FESpace&         space,
    const GaussQuadrature& quad);

/**
 * @brief Copy Host element quadrature data to Device storage.
 *
 * @param[in]     src - Host data kept alive while copies are queued.
 * @param[out]    dst - Device data replaced by the copy.
 * @param[in,out] ctx - Device execution context receiving the copies.
 */
void copy(const HostElementQuadData&            src,
          DeviceElementQuadData&                dst,
          linalg::Context<MemorySpace::Device>& ctx);

/**
 * @brief Provide non-owning access to element quadrature data.
 *
 * @tparam Space - Memory space containing the referenced arrays.
 */
template <MemorySpace Space>
class ElementQuadDataView
{
public:
  FEMX_HOST_DEVICE ElementQuadDataView() = default;

  /**
   * @brief Bind a view to flattened element quadrature arrays.
   *
   * @param[in] num_elems  - Number of elements.
   * @param[in] num_qpts   - Number of quadrature points per element.
   * @param[in] num_shapes - Number of scalar shape functions per element.
   * @param[in] dim        - Spatial dimension.
   * @param[in] N          - Shape values in quadrature-point-major order.
   * @param[in] dNdx       - Physical shape gradients in element-major order.
   * @param[in] JxW        - Weighted absolute Jacobian determinants in element-major
   * order.
   */
  FEMX_HOST_DEVICE ElementQuadDataView(
      Index                         num_elems,
      Index                         num_qpts,
      Index                         num_shapes,
      Index                         dim,
      VectorView<Space, const Real> N,
      VectorView<Space, const Real> dNdx,
      VectorView<Space, const Real> JxW)
    : num_elems_(num_elems),
      num_qpts_(num_qpts),
      num_shapes_(num_shapes),
      dim_(dim),
      N_(N),
      dNdx_(dNdx),
      JxW_(JxW)
  {
  }

  /**
   * @brief Return the number of elements.
   */
  FEMX_HOST_DEVICE Index numElems() const
  {
    return num_elems_;
  }

  /**
   * @brief Return the number of quadrature points per element.
   */
  FEMX_HOST_DEVICE Index numQuadraturePoints() const
  {
    return num_qpts_;
  }

  /**
   * @brief Return the number of scalar shape functions per element.
   */
  FEMX_HOST_DEVICE Index numShapes() const
  {
    return num_shapes_;
  }

  /**
   * @brief Return the spatial dimension.
   */
  FEMX_HOST_DEVICE Index dim() const
  {
    return dim_;
  }

  /**
   * @brief Return a shape-function value at a quadrature point.
   *
   * @param[in] iq    - Quadrature-point index.
   * @param[in] shape - Shape-function index.
   * @return Reference shape-function value.
   */
  FEMX_HOST_DEVICE Real N(Index iq, Index shape) const
  {
    return N_[iq * num_shapes_ + shape];
  }

  /**
   * @brief Return a physical shape-function gradient component.
   *
   * @param[in] ie    - Element index.
   * @param[in] iq    - Quadrature-point index.
   * @param[in] shape - Shape-function index.
   * @param[in] d     - Spatial component index.
   * @return Physical gradient component.
   */
  FEMX_HOST_DEVICE Real dNdx(Index ie,
                             Index iq,
                             Index shape,
                             Index d) const
  {
    return dNdx_[((ie * num_qpts_ + iq) * num_shapes_ + shape) * dim_ + d];
  }

  /**
   * @brief Return the weighted absolute Jacobian determinant.
   *
   * @param[in] ie - Element index.
   * @param[in] iq - Quadrature-point index.
   * @return Product of the quadrature weight and absolute Jacobian
   * determinant.
   */
  FEMX_HOST_DEVICE Real JxW(Index ie, Index iq) const
  {
    return JxW_[ie * num_qpts_ + iq];
  }

  /**
   * @brief Return the shape-function value storage.
   */
  FEMX_HOST_DEVICE const Real* NData() const
  {
    return N_.data();
  }

  /**
   * @brief Return the physical-gradient storage.
   */
  FEMX_HOST_DEVICE const Real* dNdxData() const
  {
    return dNdx_.data();
  }

  /**
   * @brief Return the weighted-Jacobian storage.
   */
  FEMX_HOST_DEVICE const Real* JxWData() const
  {
    return JxW_.data();
  }

private:
  Index                         num_elems_{0};  ///< Number of elements.
  Index                         num_qpts_{0};   ///< Quadrature points per element.
  Index                         num_shapes_{0}; ///< Scalar shapes per element.
  Index                         dim_{0};        ///< Spatial dimension.
  VectorView<Space, const Real> N_;             ///< Reference shape values.
  VectorView<Space, const Real> dNdx_;          ///< Physical shape gradients.
  VectorView<Space, const Real> JxW_;           ///< Weighted Jacobian determinants.
};

/**
 * @brief Own reusable element quadrature data in one memory space.
 *
 * @tparam Space - Memory space containing the owned arrays.
 */
template <MemorySpace Space>
class ElementQuadData
{
public:
  ElementQuadData() = default;

  ElementQuadData(const ElementQuadData&)                = default;
  ElementQuadData(ElementQuadData&&) noexcept            = default;
  ElementQuadData& operator=(const ElementQuadData&)     = default;
  ElementQuadData& operator=(ElementQuadData&&) noexcept = default;

  /**
   * @brief Return the number of elements.
   */
  Index numElems() const noexcept
  {
    return num_elems_;
  }

  /**
   * @brief Return the number of quadrature points per element.
   */
  Index numQuadraturePoints() const noexcept
  {
    return num_qpts_;
  }

  /**
   * @brief Return the number of scalar shape functions per element.
   */
  Index numShapes() const noexcept
  {
    return num_shapes_;
  }

  /**
   * @brief Return the spatial dimension.
   */
  Index dim() const noexcept
  {
    return dim_;
  }

  /**
   * @brief Return a non-owning view valid while this object is alive.
   */
  ElementQuadDataView<Space> view() const noexcept
  {
    return {num_elems_,
            num_qpts_,
            num_shapes_,
            dim_,
            N_.view(),
            dNdx_.view(),
            JxW_.view()};
  }

private:
  friend HostElementQuadData makeElementQuadData(
      const FESpace&         space,
      const GaussQuadrature& quad);

  friend void copy(const HostElementQuadData&            src,
                   DeviceElementQuadData&                dst,
                   linalg::Context<MemorySpace::Device>& ctx);

  Index               num_elems_{0};  ///< Number of elements.
  Index               num_qpts_{0};   ///< Quadrature points per element.
  Index               num_shapes_{0}; ///< Scalar shapes per element.
  Index               dim_{0};        ///< Spatial dimension.
  Vector<Space, Real> N_;             ///< Reference shape values.
  Vector<Space, Real> dNdx_;          ///< Physical shape gradients.
  Vector<Space, Real> JxW_;           ///< Weighted Jacobian determinants.
};

inline void copy(const HostElementQuadData&            src,
                 DeviceElementQuadData&                dst,
                 linalg::Context<MemorySpace::Device>& ctx)
{
  auto& vec_handler = ctx.vectorHandler();
  dst.num_elems_    = src.num_elems_;
  dst.num_qpts_     = src.num_qpts_;
  dst.num_shapes_   = src.num_shapes_;
  dst.dim_          = src.dim_;
  vec_handler.copy(src.N_, dst.N_);
  vec_handler.copy(src.dNdx_, dst.dNdx_);
  vec_handler.copy(src.JxW_, dst.JxW_);
}

} // namespace fem
} // namespace femx
