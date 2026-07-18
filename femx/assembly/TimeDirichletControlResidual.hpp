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
class MatrixOperator;
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
  TimeDirichletControlResidual(const state::TimeResidual& base,
                               fem::DirichletControl      ctr,
                               Array<Index>               fdofs             = {},
                               Index                      ctr_param_offset  = 0,
                               Index                      num_param         = -1,
                               HostVector                 fvals             = {},
                               Array<LinearInterpolation> ctr_time_stencils = {});

  state::TimeDims dims() const override;

  Index numSteps() const;
  Index numStates() const;
  Index numParams() const;
  Index numRes() const;

  void res(const state::TimeContext& ctx,
           HostVector&               out) const override;

  void applyJac(const state::TimeContext& ctx,
                state::VariableBlock      wrt,
                const HostVector&         dir,
                HostVector&               out) const override;

  void applyJacT(const state::TimeContext& ctx,
                 state::VariableBlock      wrt,
                 const HostVector&         adj,
                 HostVector&               out) const override;

  bool assembleJac(const state::TimeContext& ctx,
                   state::VariableBlock      wrt,
                   linalg::MatrixOperator&   out) const override;

  void prepareLinearSolve(const state::TimeContext& ctx,
                          state::VariableBlock      wrt,
                          linalg::MatrixOperator&   J,
                          HostVector&               rhs) const override;

  const fem::DirichletControl& control() const;

private:
  void initializeControlTimeStencils(Array<LinearInterpolation> ctr_time_stencils);

  void checkContext(const state::TimeContext& ctx) const;
  void checkVector(const HostVector* value, Index size) const;

  void replaceStateRows(linalg::MatrixOperator& out, Real diag) const;
  void eliminateStateColumns(linalg::MatrixOperator& J,
                             HostVector&             rhs) const;

  void applyControlParamJac(const state::TimeContext& ctx,
                            const HostVector&         dir,
                            HostVector&               out) const;
  void applyControlParamJacT(const state::TimeContext& ctx,
                             const HostVector&         adj,
                             HostVector&               out) const;

  HostVector interpolatedControl(
      Index             step,
      const HostVector& prm) const;
  Index ctrIndex(Index level, Index i) const;
  Real  fixedValue(Index step, Index i) const;

  Array<Index> bcRows() const;

private:
  const state::TimeResidual& base_;
  fem::DirichletControl      ctr_;
  Array<Index>               fdofs_;
  HostVector                 fvals_;
  HostVector                 base_prm_;
  state::TimeDims            base_dims_;
  state::TimeDims            dims_;
  Index                      ctr_param_offset_{0};
  Array<LinearInterpolation> ctr_time_stencils_;
  Index                      ctr_param_levels_{0};
};

} // namespace assembly
} // namespace femx
