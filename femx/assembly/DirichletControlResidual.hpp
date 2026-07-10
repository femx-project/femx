#pragma once

#include <femx/common/Types.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/state/Residual.hpp>

namespace femx
{
namespace linalg
{
class MatrixBuilder;
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
                           DirichletControl       ctr,
                           Vector<Index>          fdofs            = {},
                           Index                  ctr_param_offset = 0,
                           Index                  num_params       = -1,
                           Vector<Real>           fvals            = {});

  state::Dimensions dims() const override;

  const DirichletControl& control() const;

  void res(const Vector<Real>& state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override;

  void linearize(const Vector<Real>&   state,
                 const Vector<Real>&   prm,
                 state::Linearization& out) const override;

private:
  void checkVectorSizes(const Vector<Real>& state,
                        const Vector<Real>& prm) const;

  void replaceStateRows(linalg::MatrixBuilder& out, Real diag) const;
  void assembleParamJac(linalg::MatrixBuilder& out) const;

  Real  fixedValue(Index i) const;
  Index ctrIndex(Index i) const;

  Vector<Index> constrainedRows() const;

private:
  const state::Residual& base_;                ///< Wrapped residual before row replacement.
  DirichletControl       ctr_;                 ///< Parameter-controlled Dirichlet dofs.
  Vector<Index>          fdofs_;               ///< Fixed Dirichlet dofs.
  Vector<Real>           fvals_;               ///< Fixed Dirichlet values.
  Vector<Real>           base_prm_;            ///< Parameter vector passed to the base residual.
  state::Dimensions      base_dims_;           ///< Dimensions of the wrapped residual.
  state::Dimensions      dims_;                ///< Dimensions exposed by this wrapper.
  Index                  ctr_param_offset_{0}; ///< Offset of control parameters.
};

} // namespace assembly
} // namespace femx
