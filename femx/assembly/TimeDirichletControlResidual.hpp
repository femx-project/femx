#pragma once

#include <femx/common/LinearInterpolation.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/TimeResidual.hpp>

namespace femx
{
namespace linalg
{
class MatrixBuilder;
} // namespace linalg

namespace assembly
{

/**
 * @brief Applies time-dependent Dirichlet control rows to a TimeResidual.
 *
 * This wrapper replaces constrained state rows and exposes time-varying
 * Dirichlet values through additional control parameters.
 */
class TimeDirichletControlResidual final : public state::TimeResidual
{
public:
  TimeDirichletControlResidual(const state::TimeResidual&  base,
                               DirichletControl            ctr,
                               Vector<Index>               fdofs             = {},
                               Index                       ctr_param_offset  = 0,
                               Index                       num_params        = -1,
                               Vector<Real>                fvals             = {},
                               Vector<LinearInterpolation> ctr_time_stencils = {});

  state::TimeDims dims() const override;

  Index numSteps() const;
  Index numStates() const;
  Index numParams() const;
  Index numRes() const;

  void res(const state::TimeContext& ctx,
           Vector<Real>&             out) const override;

  void applyJac(const state::TimeContext& ctx,
                state::VariableBlock      wrt,
                const Vector<Real>&       dir,
                Vector<Real>&             out) const override;

  void applyJacT(const state::TimeContext& ctx,
                 state::VariableBlock      wrt,
                 const Vector<Real>&       adj,
                 Vector<Real>&             out) const override;

  bool assembleJac(const state::TimeContext& ctx,
                   state::VariableBlock      wrt,
                   linalg::MatrixBuilder&    out) const override;

  void prepareLinearSolve(const state::TimeContext& ctx,
                          state::VariableBlock      wrt,
                          linalg::MatrixBuilder&    J,
                          Vector<Real>&             rhs) const override;

  const DirichletControl& control() const;

private:
  void initializeControlTimeStencils(Vector<LinearInterpolation> ctr_time_stencils);

  void checkContext(const state::TimeContext& ctx) const;
  void checkVector(const Vector<Real>* value, Index size) const;

  void replaceStateRows(linalg::MatrixBuilder& out, Real diag) const;
  void eliminateStateColumns(linalg::MatrixBuilder& J,
                             Vector<Real>&          rhs) const;

  void applyControlParamJac(const state::TimeContext& ctx,
                            const Vector<Real>&       dir,
                            Vector<Real>&             out) const;
  void applyControlParamJacT(const state::TimeContext& ctx,
                             const Vector<Real>&       adj,
                             Vector<Real>&             out) const;

  Real  ctrValue(Index step, Index i, const Vector<Real>& prm) const;
  Index ctrIndex(Index level, Index i) const;
  Real  fixedValue(Index step, Index i) const;

  Vector<Index> constrainedRows() const;

private:
  const state::TimeResidual&  base_;
  DirichletControl            ctr_;
  Vector<Index>               fdofs_;
  Vector<Real>                fvals_;
  Vector<Real>                base_prm_;
  state::TimeDims             base_dims_;
  state::TimeDims             dims_;
  Index                       ctr_param_offset_{0};
  Vector<LinearInterpolation> ctr_time_stencils_;
  Index                       ctr_param_levels_{0};
};

} // namespace assembly
} // namespace femx
