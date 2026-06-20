#pragma once

#include <femx/algebra/MatrixBuilder.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/core/Types.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/problem/TimeResidual.hpp>

namespace femx
{
namespace assembly
{

/** @brief Applies time-dependent Dirichlet control rows to a TimeResidual. */
class TimeDirichletControlResidual final : public problem::TimeResidual
{
public:
  TimeDirichletControlResidual(const problem::TimeResidual& base,
                               DirichletControl             control,
                               Vector<Index>                fixed_dofs           = {},
                               Index                        control_param_offset = 0,
                               Index                        num_params           = -1,
                               Vector<Real>                 fixed_values         = {});

  problem::TimeDimensions dimensions() const override;

  Index numSteps() const;
  Index numStates() const;
  Index numParams() const;
  Index numRes() const;

  void residual(const problem::TimeContext& ctx,
                Vector<Real>&               out) const override;

  void applyJacobian(const problem::TimeContext& ctx,
                     problem::VariableBlock      wrt,
                     const Vector<Real>&         dir,
                     Vector<Real>&               out) const override;

  void applyJacobianT(const problem::TimeContext& ctx,
                      problem::VariableBlock      wrt,
                      const Vector<Real>&         adjoint,
                      Vector<Real>&               out) const override;

  bool assembleJacobian(const problem::TimeContext& ctx,
                        problem::VariableBlock      wrt,
                        algebra::MatrixBuilder&     out) const override;

  const DirichletControl& control() const;

private:
  void checkContext(const problem::TimeContext& ctx) const;
  void checkVector(const Vector<Real>* value, Index size, const char* name) const;

  void replaceStateRows(algebra::MatrixBuilder& out, Real diag) const;

  void applyControlParamJac(const problem::TimeContext& ctx,
                            const Vector<Real>&         dir,
                            Vector<Real>&               out) const;
  void applyControlParamJacT(const problem::TimeContext& ctx,
                             const Vector<Real>&         adjoint,
                             Vector<Real>&               out) const;

  Index controlParamIndex(Index step, Index i) const;
  Real  fixedValue(Index step, Index i) const;

  Vector<Index> constrainedRows() const;

private:
  const problem::TimeResidual& base_;
  DirichletControl             control_;
  Vector<Index>                fixed_dofs_;
  Vector<Real>                 fixed_values_;
  Vector<Real>                 base_prm_;
  problem::TimeDimensions      base_dims_;
  problem::TimeDimensions      dims_;
  Index                        control_param_offset_{0};
};

} // namespace assembly
} // namespace femx
