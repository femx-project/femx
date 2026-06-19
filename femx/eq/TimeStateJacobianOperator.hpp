#pragma once

#include <femx/common/Types.hpp>
#include <femx/eq/TimeResidualEquation.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/LinearOperator.hpp>

namespace femx
{
namespace eq
{

enum class TimeStateSlot
{
  Previous,
  Next
};

/** @brief Matrix-free state Jacobian operator for one time-step residual. */
class TimeStateJacobianOperator final : public system::LinearOperator
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

} // namespace eq
} // namespace femx
