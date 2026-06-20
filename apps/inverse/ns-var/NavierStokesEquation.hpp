#pragma once

#include "Components.hpp"
#include <femx/algebra/MatrixBuilder.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/core/Types.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/fem/Quadrature.hpp>
#include <femx/problem/TimeResidual.hpp>

namespace femx
{

struct TimeNavierStokesParameters
{
  Index       steps = 1;
  Real        dt    = 1.0;
  FluidParams fluid;
  Index       quad_order = 2;
};

/** @brief Semi-implicit GLS time residual for incompressible Navier-Stokes.
 *
 * The mixed space is expected to contain velocity as field 0 and pressure as
 * field 1. This app-local implementation supports equal velocity/pressure
 * scalar element shapes on the same mesh.
 */
class NavierStokesEquation final : public problem::TimeResidual
{
public:
  NavierStokesEquation(const MixedFESpace&        space,
                       TimeNavierStokesParameters parameters);

  void setCellRange(Index begin, Index end);

  problem::TimeDimensions dimensions() const override;

  Index numSteps() const;
  Index numStates() const;
  Index numParams() const;
  Index numRes() const;

  void residual(const problem::TimeContext& ctx,
                Vector<Real>& out) const override;

  void applyJacobian(const problem::TimeContext& ctx,
                     problem::VariableBlock wrt,
                     const Vector<Real>& dir,
                     Vector<Real>& out) const override;

  void applyJacobianT(const problem::TimeContext& ctx,
                      problem::VariableBlock wrt,
                      const Vector<Real>& adjoint,
                      Vector<Real>& out) const override;

  bool assembleJacobian(const problem::TimeContext& ctx,
                        problem::VariableBlock wrt,
                        algebra::MatrixBuilder& out) const override;

  void res(Index               step,
           const Vector<Real>& x_next,
           const Vector<Real>& x,
           const Vector<Real>& prm,
           Vector<Real>&       out) const;

  void assembleNextStateJac(Index                 step,
                            const Vector<Real>&   x_next,
                            const Vector<Real>&   x,
                            const Vector<Real>&   prm,
                            algebra::MatrixBuilder& out) const;

  void assemblePrevStateJac(Index                 step,
                            const Vector<Real>&   x_next,
                            const Vector<Real>&   x,
                            const Vector<Real>&   prm,
                            algebra::MatrixBuilder& out) const;

  void assembleParamJac(Index                 step,
                        const Vector<Real>&   x_next,
                        const Vector<Real>&   x,
                        const Vector<Real>&   prm,
                        algebra::MatrixBuilder& out) const;

private:
  void checkSizes(Index               step,
                  const Vector<Real>& x_next,
                  const Vector<Real>& x,
                  const Vector<Real>& prm) const;

  void checkSpace() const;
  void checkParameters() const;

private:
  const MixedFESpace&        space_;
  TimeNavierStokesParameters prm_;
  GaussQuadrature            quad_;
  Index                      cell_begin_{0};
  Index                      cell_end_{0};
};

} // namespace femx
