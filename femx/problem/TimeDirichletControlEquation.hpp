#pragma once

#include <femx/core/Types.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/problem/TimeMatrixResidualEquation.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/SystemMatrix.hpp>

namespace femx
{
namespace problem
{

class TimeDirichletControlEquation final : public TimeMatrixResidualEquation
{
public:
  TimeDirichletControlEquation(
      const TimeMatrixResidualEquation&     base_eq,
      DirichletControl                      control,
      Vector<Index>                         fixed_dofs           = {},
      Index                                 control_param_offset = 0,
      Index                                 num_params           = -1,
      Vector<Real>                          fixed_values         = {});

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

  void applyParamJac(Index               step,
                     const Vector<Real>& x_next,
                     const Vector<Real>& x,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override;

  void applyParamJacT(Index               step,
                      const Vector<Real>& x_next,
                      const Vector<Real>& x,
                      const Vector<Real>& prm,
                      const Vector<Real>& lambda,
                      Vector<Real>&       out) const override;

  const DirichletControl& control() const;

private:
  void checkSizes(Index               step,
                  const Vector<Real>& x_next,
                  const Vector<Real>& x,
                  const Vector<Real>& prm) const;

  void replaceStateRows(algebra::SystemMatrix& out,
                        Real                  diag) const;

  Index controlParamIndex(Index step,
                          Index i) const;

  Real fixedValue(Index step,
                  Index i) const;

private:
  const TimeMatrixResidualEquation&     base_eq_;
  DirichletControl                      ctr_;
  Vector<Index>                         fixed_dofs_;
  Vector<Real>                          fixed_values_;
  Vector<Real>                          base_prm_;
  Index                                 control_param_offset_{0};
  Index                                 num_params_{0};
};

} // namespace problem
} // namespace femx
