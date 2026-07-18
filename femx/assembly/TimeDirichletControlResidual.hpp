#pragma once

#include <femx/common/LinearInterpolation.hpp>
#include <femx/common/Types.hpp>
#include <femx/fem/ControlMap.hpp>
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
 * @brief Host TimeResidual adapter for fixed and controlled Dirichlet rows.
 *
 * Control interpolation, fixed values, and parameter products are owned by one
 * HostControlMap. The adapter retains preallocated boundary work and is not
 * reentrant.
 */
class TimeDirichletControlResidual final : public state::TimeResidual
{
public:
  /** @brief Wrap a Host residual with fixed and controlled Dirichlet rows. */
  TimeDirichletControlResidual(
      const state::TimeResidual& base,
      fem::DirichletControl      ctr,
      Array<Index>               fixed_dofs = {},
      Index                      ctr_off    = 0,
      Index                      num_prm    = -1,
      HostVector                 fixed_vals = {},
      Array<LinearInterpolation> time       = {});

  /** @brief Return the wrapped time dimensions including parameters. */
  state::TimeDims dims() const override;

  /** @brief Return the number of residual time steps. */
  Index numSteps() const;
  /** @brief Return the number of state entries at one time level. */
  Index numStates() const;
  /** @brief Return the full parameter-vector size. */
  Index numParams() const;
  /** @brief Return the residual-vector size. */
  Index numRes() const;

  /** @brief Evaluate the row-replaced residual. */
  void res(const state::TimeContext& ctx,
           HostVector&               out) const override;

  /** @brief Apply one Jacobian block to `dir`. */
  void applyJac(const state::TimeContext& ctx,
                state::VariableBlock      wrt,
                const HostVector&         dir,
                HostVector&               out) const override;

  /** @brief Apply one transposed Jacobian block to `adj`. */
  void applyJacT(const state::TimeContext& ctx,
                 state::VariableBlock      wrt,
                 const HostVector&         adj,
                 HostVector&               out) const override;

  /** @brief Assemble one row-replaced Jacobian block when supported. */
  bool assembleJac(const state::TimeContext& ctx,
                   state::VariableBlock      wrt,
                   linalg::MatrixOperator&   out) const override;

  /** @brief Eliminate constrained columns in the forward solve system. */
  void prepareLinearSolve(const state::TimeContext& ctx,
                          state::VariableBlock      wrt,
                          linalg::MatrixOperator&   jac,
                          HostVector&               rhs) const override;

private:
  void checkContext(const state::TimeContext& ctx) const;
  void replaceRows(linalg::MatrixOperator& mat, Real diag) const;
  void eliminateCols(linalg::MatrixOperator& mat,
                     HostVector&             rhs) const;

private:
  const state::TimeResidual& base_;
  fem::HostControlMap        ctr_;
  Array<Index>               rows_;
  HostVector                 base_prm_;
  mutable HostVector         bc_vals_;
  state::TimeDims            base_dims_;
  state::TimeDims            dims_;
};

} // namespace assembly
} // namespace femx
