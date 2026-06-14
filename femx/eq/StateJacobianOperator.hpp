#pragma once

#include <femx/common/Types.hpp>
#include <femx/system/LinearOperator.hpp>
#include <femx/eq/ResidualEquation.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace equation
{

/** @brief LinearOperator view of R_u(u,m). */
class StateJacobianOperator final : public system::LinearOperator
{
public:
  StateJacobianOperator(const ResidualEquation& equation,
                        const Vector&           state,
                        const Vector&           params)
    : equation_(equation),
      state_(state),
      params_(params)
  {
  }

  index_type numRows() const override
  {
    return equation_.numResiduals();
  }

  index_type numCols() const override
  {
    return equation_.numStates();
  }

  void apply(const Vector& dir, Vector& out) const override
  {
    equation_.applyStateJac(state_, params_, dir, out);
  }

  void applyT(const Vector& dir, Vector& out) const override
  {
    equation_.applyStateJacT(state_, params_, dir, out);
  }

private:
  const ResidualEquation& equation_;
  const Vector&           state_;
  const Vector&           params_;
};

} // namespace equation
} // namespace femx
