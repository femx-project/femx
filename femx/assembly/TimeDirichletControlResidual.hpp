#pragma once

#include <femx/common/LinearInterpolation.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/linalg/MatrixBuilder.hpp>
#include <femx/linalg/Vector.hpp>
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
                               Vector<Index>                fixed_dofs            = {},
                               Index                        ctr_param_offset  = 0,
                               Index                        num_params            = -1,
                               Vector<Real>                 fixed_values          = {},
                               Vector<LinearInterpolation>  ctr_time_stencils = {});

  problem::TimeDims dimensions() const override;

  Index numSteps() const;
  Index numStates() const;
  Index numParams() const;
  Index numRes() const;

  void residual(const problem::TimeContext& ctx,
                Vector<Real>&               out) const override;

  void applyJac(const problem::TimeContext& ctx,
                problem::VariableBlock      wrt,
                const Vector<Real>&         dir,
                Vector<Real>&               out) const override;

  void applyJacT(const problem::TimeContext& ctx,
                 problem::VariableBlock      wrt,
                 const Vector<Real>&         adjoint,
                 Vector<Real>&               out) const override;

  bool assembleJacobian(const problem::TimeContext& ctx,
                        problem::VariableBlock      wrt,
                        linalg::MatrixBuilder&      out) const override;

  const DirichletControl& control() const;

private:
  void initializeControlTimeStencils(
      Vector<LinearInterpolation> ctr_time_stencils);

  void checkContext(const problem::TimeContext& ctx) const;
  void checkVector(const Vector<Real>* value, Index size, const char* name) const;

  void replaceStateRows(linalg::MatrixBuilder& out, Real diag) const;

  void applyControlParamJac(const problem::TimeContext& ctx,
                            const Vector<Real>&         dir,
                            Vector<Real>&               out) const;
  void applyControlParamJacT(const problem::TimeContext& ctx,
                             const Vector<Real>&         adjoint,
                             Vector<Real>&               out) const;

  Real  ctrValue(Index step, Index i, const Vector<Real>& prm) const;
  Index ctrIndex(Index level, Index i) const;
  Real  fixedValue(Index step, Index i) const;

  Vector<Index> constrainedRows() const;

private:
  const problem::TimeResidual& base_;
  DirichletControl             ctr_;
  Vector<Index>                fixed_dofs_;
  Vector<Real>                 fixed_values_;
  Vector<Real>                 base_prm_;
  problem::TimeDims      base_dims_;
  problem::TimeDims      dims_;
  Index                        ctr_param_offset_{0};
  Vector<LinearInterpolation>  ctr_time_stencils_;
  Index                        ctr_param_levels_{0};
};

} // namespace assembly
} // namespace femx
