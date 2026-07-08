#pragma once

#include <femx/common/Types.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/MatrixBuilder.hpp>
#include <femx/problem/Residual.hpp>

namespace femx
{
namespace assembly
{

/**
 * @brief Applies stationary Dirichlet control rows to a Residual.
 *
 * This wrapper replaces selected residual rows with either fixed Dirichlet
 * values or parameter-controlled values, and updates the state/parameter
 * Jacobians consistently.
 */
class DirichletControlResidual final : public problem::Residual
{
public:
  DirichletControlResidual(const problem::Residual& base,
                           DirichletControl         ctr,
                           Vector<Index>            fdofs            = {},
                           Index                    ctr_param_offset = 0,
                           Index                    num_params       = -1,
                           Vector<Real>             fvals            = {});

  problem::Dimensions dims() const override;

  const DirichletControl& control() const;

  void res(const Vector<Real>& state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override;

  void linearize(const Vector<Real>&     state,
                 const Vector<Real>&     prm,
                 problem::Linearization& out) const override;

private:
  void checkVectorSizes(const Vector<Real>& state,
                        const Vector<Real>& prm) const;

  void replaceStateRows(linalg::MatrixBuilder& out, Real diag) const;
  void assembleParamJac(linalg::MatrixBuilder& out) const;

  Real  fixedValue(Index i) const;
  Index ctrIndex(Index i) const;

  Vector<Index> constrainedRows() const;

private:
  const problem::Residual& base_;
  DirichletControl         ctr_;
  Vector<Index>            fdofs_;
  Vector<Real>             fvals_;
  Vector<Real>             base_prm_;
  problem::Dimensions      base_dims_;
  problem::Dimensions      dims_;
  Index                    ctr_param_offset_{0};
};

} // namespace assembly
} // namespace femx
