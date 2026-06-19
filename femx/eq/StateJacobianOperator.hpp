#pragma once

#include <femx/common/Types.hpp>
#include <femx/eq/ResidualEquation.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/LinearOperator.hpp>

namespace femx
{
namespace eq
{

/** @brief LinearOperator view of R_u(u,m). */
class StateJacobianOperator final : public system::LinearOperator
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

} // namespace eq
} // namespace femx
