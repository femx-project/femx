#pragma once

#include <vector>

#include <femx/core/Math.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/problem/TimeObservation.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{
namespace fem
{

using problem::TimeObservation;

/** @brief Samples field components at physical points on each time level. */
class TimePointSampler final : public TimeObservation
{
public:
  TimePointSampler(Index               num_steps,
                   const MixedFESpace& space,
                   Index               field_id,
                   std::vector<Point3> points,
                   Vector<Index>       components,
                   Index               num_prm);

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

  const std::vector<Point3>& points() const;

  const Vector<Index>& components() const;

  static bool containsPoint(const MixedFESpace& space,
                            Index               field_id,
                            const Point3&       point);

  static std::vector<Point3> filterPointsInside(
      const MixedFESpace&      space,
      Index                    field_id,
      const std::vector<Point3>& points);

private:
  struct Stencil
  {
    Vector<Index> indices;
    Vector<Real>  weights;
  };

  void checkLevel(Index level) const;

  void checkInputs(const Vector<Real>& state,
                   const Vector<Real>& prm) const;

  static std::vector<Stencil> buildStencils(
      const MixedFieldView&      field,
      const std::vector<Point3>& points,
      const Vector<Index>&       components);

  static void resize(Vector<Real>& out,
                     Index         size);

private:
  Index                num_steps_{0};
  Index                num_states_{0};
  Index                num_prm_{0};
  std::vector<Point3>  points_;
  Vector<Index>        components_;
  std::vector<Stencil> stencils_;
};

} // namespace fem
} // namespace femx
