#pragma once

#include <femx/common/Math.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/inverse/TimeObservationOperator.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace fem
{

using inverse::TimeObservationOperator;

/**
 * @brief Interpolates field components at physical points on each time level.
 *
 * TimePointInterpolator precomputes point stencils and applies observation
 * Jacobian products for time-dependent inverse problems.
 */
class TimePointInterpolator final : public TimeObservationOperator
{
public:
  TimePointInterpolator(Index               num_steps,
                        const MixedFESpace& space,
                        Index               fid,
                        Array<Point3>       pts,
                        Array<Index>        comps,
                        Index               num_param);

  Index numSteps() const override;

  Index numStates() const override;

  Index numParams() const override;

  Index numObservations() const override;

  void observe(Index             level,
               const HostVector& state,
               const HostVector& prm,
               HostVector&       out) const override;

  void applyStateJac(Index             level,
                     const HostVector& state,
                     const HostVector& prm,
                     const HostVector& dir,
                     HostVector&       out) const override;

  void applyStateJacT(Index             level,
                      const HostVector& state,
                      const HostVector& prm,
                      const HostVector& dir,
                      HostVector&       out) const override;

  void applyParamJac(Index             level,
                     const HostVector& state,
                     const HostVector& prm,
                     const HostVector& dir,
                     HostVector&       out) const override;

  void applyParamJacT(Index             level,
                      const HostVector& state,
                      const HostVector& prm,
                      const HostVector& dir,
                      HostVector&       out) const override;

  const Array<Point3>& pts() const;

  const Array<Index>& comps() const;

  static bool containsPoint(const MixedFESpace& space,
                            Index               fid,
                            const Point3&       point);

  static Array<Point3> filterPointsInside(
      const MixedFESpace&  space,
      Index                fid,
      const Array<Point3>& pts);

private:
  struct Stencil
  {
    Array<Index> indices; ///< State dofs used by one observation.
    HostVector   wts;     ///< Interpolation weights for those dofs.
  };

  void checkLevel(Index level) const;

  void checkInputs(const HostVector& state,
                   const HostVector& prm) const;

  static Array<Stencil> buildStencils(
      const MixedFieldView& field,
      const Array<Point3>&  pts,
      const Array<Index>&   comps);

private:
  Index          num_steps_{0};
  Index          num_states_{0};
  Index          num_param_{0};
  Array<Point3>  pts_;      ///< Observation point coordinates.
  Array<Index>   comps_;    ///< Observed component at each point.
  Array<Stencil> stencils_; ///< Interpolation stencil for each point.
};

} // namespace fem
} // namespace femx
