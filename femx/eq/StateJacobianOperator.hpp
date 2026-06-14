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
  StateJacobianOperator(const ResidualEquation& equation,
                        const Vector&           state,
                        const Vector&           params)
    : eq_(equation),
      state_(state),
      params_(params)
  {
  }

  Index numRows() const override
  {
    return eq_.numRes();
  }

  Index numCols() const override
  {
    return eq_.numStates();
  }

  void apply(const Vector& dir, Vector& out) const override
  {
    eq_.applyStateJac(state_, params_, dir, out);
  }

  void applyT(const Vector& dir, Vector& out) const override
  {
    eq_.applyStateJacT(state_, params_, dir, out);
  }

private:
  const ResidualEquation& eq_;
  const Vector&           state_;
  const Vector&           params_;
};

} // namespace eq
} // namespace femx
