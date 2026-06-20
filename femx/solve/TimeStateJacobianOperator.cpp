#include <femx/solve/TimeStateJacobianOperator.hpp>

namespace femx
{
namespace solve
{

TimeStateJacobianOperator::TimeStateJacobianOperator(
    const TimeResidualEquation& eq,
    Index                       step,
    const Vector<Real>&         x_next,
    const Vector<Real>&         x,
    const Vector<Real>&         prm,
    TimeStateSlot               slot)
  : eq_(eq),
    step_(step),
    x_next_(x_next),
    x_(x),
    prm_(prm),
    slot_(slot)
{
}

Index TimeStateJacobianOperator::numRows() const
{
  return eq_.numRes();
}

Index TimeStateJacobianOperator::numCols() const
{
  return eq_.numStates();
}

void TimeStateJacobianOperator::apply(const Vector<Real>& dir,
                                      Vector<Real>&       out) const
{
  if (slot_ == TimeStateSlot::Next)
  {
    eq_.applyNextStateJac(step_, x_next_, x_, prm_, dir, out);
  }
  else
  {
    eq_.applyPreviousStateJac(step_, x_next_, x_, prm_, dir, out);
  }
}

void TimeStateJacobianOperator::applyT(const Vector<Real>& dir,
                                       Vector<Real>&       out) const
{
  if (slot_ == TimeStateSlot::Next)
  {
    eq_.applyNextStateJacT(step_, x_next_, x_, prm_, dir, out);
  }
  else
  {
    eq_.applyPreviousStateJacT(step_, x_next_, x_, prm_, dir, out);
  }
}

} // namespace solve
} // namespace femx
