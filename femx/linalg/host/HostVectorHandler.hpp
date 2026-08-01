#pragma once

#include <femx/linalg/VectorHandler.hpp>

namespace femx::linalg
{

/**
 * @brief Provide Host vector operations.
 */
class HostVectorHandler final : public VectorHandler<MemorySpace::Host>
{
public:
  /**
   * @brief Set every value to zero.
   *
   * @param[out] vals - Values to clear.
   */
  void zero(HostVectorView<Real> vals) const override;

  /**
   * @brief Compute `y = a * x + b * y`.
   *
   * @param[in]     a - Input-vector scale.
   * @param[in]     x - Input vector.
   * @param[in]     b - Existing-output scale.
   * @param[in,out] y - Output vector.
   * @throws - If sizes or storage overlap are invalid.
   */
  void axpby(Real                       a,
             HostVectorView<const Real> x,
             Real                       b,
             HostVectorView<Real>       y) const override;

  /**
   * @brief Compute the dot product of two vectors.
   *
   * @param[in] x - First input vector.
   * @param[in] y - Second input vector.
   * @return Dot product of `x` and `y`.
   * @throws - If vector sizes differ.
   */
  Real dot(HostVectorView<const Real> x,
           HostVectorView<const Real> y) const override;

  /**
   * @brief Gather indexed source values into a contiguous destination.
   *
   * @param[in]  src - Source values.
   * @param[in]  indices - Source indices in destination order.
   * @param[out] dst - Contiguous destination values.
   * @throws - If sizes, indices, or aliasing are invalid.
   */
  void gather(HostVectorView<const Real>  src,
              HostVectorView<const Index> indices,
              HostVectorView<Real>        dst) const override;

  /**
   * @brief Scatter contiguous source values to indexed destinations.
   *
   * @param[in]  src - Contiguous source values.
   * @param[in]  indices - Destination indices in source order.
   * @param[out] dst - Indexed destination values.
   * @throws - If sizes, indices, or aliasing are invalid.
   */
  void scatter(HostVectorView<const Real>  src,
               HostVectorView<const Index> indices,
               HostVectorView<Real>        dst) const override;
};

} // namespace femx::linalg
