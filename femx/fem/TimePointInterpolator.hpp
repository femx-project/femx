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
                        Vector<Point3>      pts,
                        Vector<Index>       comps,
                        Index               num_param);

  Index numSteps() const override;

  Index numStates() const override;

  Index numParams() const override;

  Index numObservations() const override;

  void observe(Index               level,
               const Vector<Real>& state,
               const Vector<Real>& prm,
               Vector<Real>&       out) const override;

  void applyStateJac(Index               level,
                     const Vector<Real>& state,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override;

  void applyStateJacT(Index               level,
                      const Vector<Real>& state,
                      const Vector<Real>& prm,
                      const Vector<Real>& dir,
                      Vector<Real>&       out) const override;

  void applyParamJac(Index               level,
                     const Vector<Real>& state,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override;

  void applyParamJacT(Index               level,
                      const Vector<Real>& state,
                      const Vector<Real>& prm,
                      const Vector<Real>& dir,
                      Vector<Real>&       out) const override;

  const Vector<Point3>& pts() const;

  const Vector<Index>& comps() const;

  static bool containsPoint(const MixedFESpace& space,
                            Index               fid,
                            const Point3&       point);

  static Vector<Point3> filterPointsInside(
      const MixedFESpace&   space,
      Index                 fid,
      const Vector<Point3>& pts);

private:
  struct Stencil
  {
    Vector<Index> indices; ///< State dofs used by one observation.
    Vector<Real>  wts;     ///< Interpolation weights for those dofs.
  };

  void checkLevel(Index level) const;

  void checkInputs(const Vector<Real>& state,
                   const Vector<Real>& prm) const;

  static Vector<Stencil> buildStencils(
      const MixedFieldView& field,
      const Vector<Point3>& pts,
      const Vector<Index>&  comps);

private:
  Index           num_steps_{0};
  Index           num_states_{0};
  Index           num_param_{0};
  Vector<Point3>  pts_;      ///< Observation point coordinates.
  Vector<Index>   comps_;    ///< Observed component at each point.
  Vector<Stencil> stencils_; ///< Interpolation stencil for each point.
};

} // namespace fem
} // namespace femx
