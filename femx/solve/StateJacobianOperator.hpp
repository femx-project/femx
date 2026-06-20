#pragma once

#include <femx/core/Types.hpp>
#include <femx/problem/ResidualEquation.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/LinearOperator.hpp>

namespace femx
{
namespace solve
{

using problem::ResidualEquation;

/** @brief LinearOperator view of R_u(u,m). */
class StateJacobianOperator final : public algebra::LinearOperator
{
public:
  StateJacobianOperator(const ResidualEquation& eq,
                        const Vector<Real>&     state,
                        const Vector<Real>&     prm);

  Index numRows() const override;

  Index numCols() const override;

  void apply(const Vector<Real>& dir, Vector<Real>& out) const override;

  void applyT(const Vector<Real>& dir, Vector<Real>& out) const override;

private:
  const ResidualEquation& eq_;
  const Vector<Real>&     state_;
  const Vector<Real>&     prm_;
};

} // namespace solve
} // namespace femx
