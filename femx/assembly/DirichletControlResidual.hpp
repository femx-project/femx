#pragma once

#include <femx/common/Types.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/Residual.hpp>

namespace femx
{
namespace linalg
{
class MatrixOperator;
} // namespace linalg

namespace assembly
{

/**
 * @brief Applies stationary Dirichlet control rows to a Residual.
 *
 * This wrapper replaces selected residual rows with either fixed Dirichlet
 * values or parameter-controlled values, and updates the state/parameter
 * Jacobians consistently.
 */
class DirichletControlResidual final : public state::Residual
{
public:
  DirichletControlResidual(const state::Residual& base,
                           fem::DirichletControl  ctr,
                           Array<Index>           fdofs            = {},
                           Index                  ctr_param_offset = 0,
                           Index                  num_param        = -1,
                           HostVector             fvals            = {});

  state::Dimensions dims() const override;

  const fem::DirichletControl& control() const;

  void res(const HostVector& state,
           const HostVector& prm,
           HostVector&       out) const override;

  void linearize(const HostVector&     state,
                 const HostVector&     prm,
                 state::Linearization& out) const override;

private:
  void checkVectorSizes(const HostVector& state,
                        const HostVector& prm) const;

  void replaceStateRows(linalg::MatrixOperator& out, Real diag) const;
  void assembleParamJac(linalg::MatrixOperator& out) const;

  HostVector controlVals(const HostVector& prm) const;
  Real       fixedValue(Index i) const;
  Index      ctrIndex(Index i) const;

  Array<Index> bcRows() const;

private:
  const state::Residual& base_;                ///< Wrapped residual before row replacement.
  fem::DirichletControl  ctr_;                 ///< Parameter-controlled Dirichlet dofs.
  Array<Index>           fdofs_;               ///< Fixed Dirichlet dofs.
  HostVector             fvals_;               ///< Fixed Dirichlet values.
  HostVector             base_prm_;            ///< Parameter vector passed to the base residual.
  state::Dimensions      base_dims_;           ///< Dimensions of the wrapped residual.
  state::Dimensions      dims_;                ///< Dimensions exposed by this wrapper.
  Index                  ctr_param_offset_{0}; ///< Offset of control parameters.
};

} // namespace assembly
} // namespace femx
