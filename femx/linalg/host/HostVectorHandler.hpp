#pragma once

#include <femx/linalg/VectorHandler.hpp>

namespace femx::linalg
{

/**
 * @brief Provide Host vector operations.
 */
class HostVectorHandler final : public VectorHandler<MemorySpace::Host>
{
  using Base = VectorHandler<MemorySpace::Host>;

public:
  /**
   * @copydoc Base::zero()
   */
  void zero(HostVectorView<Real> vals) const override;

  /**
   * @copydoc Base::axpby()
   */
  void axpby(Real                       a,
             HostVectorView<const Real> x,
             Real                       b,
             HostVectorView<Real>       y) const override;

  /**
   * @copydoc Base::dot()
   */
  Real dot(HostVectorView<const Real> x,
           HostVectorView<const Real> y) const override;

  /**
   * @copydoc Base::gather()
   */
  void gather(HostVectorView<const Real>  src,
              HostVectorView<const Index> idx,
              HostVectorView<Real>        dst) const override;

  /**
   * @copydoc Base::scatter()
   */
  void scatter(HostVectorView<const Real>  src,
               HostVectorView<const Index> idx,
               HostVectorView<Real>        dst) const override;
};

} // namespace femx::linalg
