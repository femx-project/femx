#include <femx/solve/StateJacobianOperator.hpp>

namespace femx
{
namespace solve
{

StateJacobianOperator::StateJacobianOperator(const ResidualEquation& eq,
                                             const Vector<Real>&     state,
                                             const Vector<Real>&     prm)
  : eq_(eq),
    state_(state),
    prm_(prm)
{
}

Index StateJacobianOperator::numRows() const
{
  return eq_.numRes();
}

Index StateJacobianOperator::numCols() const
{
  return eq_.numStates();
}

void StateJacobianOperator::apply(const Vector<Real>& dir,
                                  Vector<Real>&       out) const
{
  eq_.applyStateJac(state_, prm_, dir, out);
}

void StateJacobianOperator::applyT(const Vector<Real>& dir,
                                   Vector<Real>&       out) const
{
  eq_.applyStateJacT(state_, prm_, dir, out);
}

} // namespace solve
} // namespace femx
