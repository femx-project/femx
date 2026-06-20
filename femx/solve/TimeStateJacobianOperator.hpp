#pragma once

#include <femx/core/Types.hpp>
#include <femx/problem/TimeResidualEquation.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/algebra/LinearOperator.hpp>

namespace femx
{
namespace solve
{

using problem::TimeResidualEquation;

enum class TimeStateSlot
{
  Previous,
  Next
};

/** @brief Matrix-free state Jacobian operator for one time-step residual. */
class TimeStateJacobianOperator final : public algebra::LinearOperator
{
public:
  TimeStateJacobianOperator(const TimeResidualEquation& eq,
                            Index                       step,
                            const Vector<Real>&         x_next,
                            const Vector<Real>&         x,
                            const Vector<Real>&         prm,
                            TimeStateSlot               slot);

  Index numRows() const override;

  Index numCols() const override;

  void apply(const Vector<Real>& dir,
             Vector<Real>&       out) const override;

  void applyT(const Vector<Real>& dir,
              Vector<Real>&       out) const override;

private:
  const TimeResidualEquation& eq_;
  Index                       step_{0};
  const Vector<Real>&         x_next_;
  const Vector<Real>&         x_;
  const Vector<Real>&         prm_;
  TimeStateSlot               slot_{TimeStateSlot::Next};
};

} // namespace solve
} // namespace femx
