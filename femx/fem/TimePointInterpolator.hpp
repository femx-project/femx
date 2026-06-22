#pragma once

#include <femx/common/Math.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/TimeObservationOperator.hpp>

namespace femx
{
namespace fem
{

using problem::TimeObservationOperator;

/** @brief Interpolates field components at physical points on each time level. */
class TimePointInterpolator final : public TimeObservationOperator
{
public:
  TimePointInterpolator(Index               nt,
                        const MixedFESpace& space,
                        Index               fid,
                        Vector<Point3>      pts,
                        Vector<Index>       comps,
                        Index               nprm);

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
    Vector<Index> indices;
    Vector<Real>  wts;
  };

  void checkLevel(Index level) const;

  void checkInputs(const Vector<Real>& state,
                   const Vector<Real>& prm) const;

  static Vector<Stencil> buildStencils(
      const MixedFieldView& field,
      const Vector<Point3>& pts,
      const Vector<Index>&  comps);

private:
  Index           nt_{0};
  Index           nst_{0};
  Index           nprm_{0};
  Vector<Point3>  pts_;
  Vector<Index>   comps_;
  Vector<Stencil> stencils_;
};

} // namespace fem
} // namespace femx
