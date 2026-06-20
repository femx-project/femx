#pragma once

#include "Components.hpp"
#include <femx/core/Types.hpp>
#include <femx/problem/TimeMatrixResidual.hpp>
#include <femx/fem/Quadrature.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/SystemMatrix.hpp>

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
class NavierStokesEquation final : public problem::TimeMatrixResidual
{
public:
  NavierStokesEquation(const MixedFESpace&        space,
                       TimeNavierStokesParameters parameters);

  void setCellRange(Index begin, Index end);

  Index numSteps() const override;
  Index numStates() const override;
  Index numParams() const override;
  Index numRes() const override;

  void res(Index               step,
           const Vector<Real>& x_next,
           const Vector<Real>& x,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override;

  void assembleNextStateJac(Index                 step,
                            const Vector<Real>&   x_next,
                            const Vector<Real>&   x,
                            const Vector<Real>&   prm,
                            algebra::SystemMatrix& out) const override;

  void assemblePrevStateJac(Index                 step,
                            const Vector<Real>&   x_next,
                            const Vector<Real>&   x,
                            const Vector<Real>&   prm,
                            algebra::SystemMatrix& out) const override;

  void assembleParamJac(Index                 step,
                        const Vector<Real>&   x_next,
                        const Vector<Real>&   x,
                        const Vector<Real>&   prm,
                        algebra::SystemMatrix& out) const override;

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
