#include <femx/problem/TimeMatrixResidualEquation.hpp>
#include <femx/algebra/backends/native/DenseSystemMatrix.hpp>

using namespace femx::algebra;

namespace femx
{
namespace problem
{

void TimeMatrixResidualEquation::applyNextStateJac(
    Index               step,
    const Vector<Real>& x_next,
    const Vector<Real>& x,
    const Vector<Real>& prm,
    const Vector<Real>& dir,
    Vector<Real>&       out) const
{
  DenseSystemMatrix jac;
  assembleNextStateJac(step, x_next, x, prm, jac);
  jac.finalize();
  jac.apply(dir, out);
}

void TimeMatrixResidualEquation::applyNextStateJacT(
    Index               step,
    const Vector<Real>& x_next,
    const Vector<Real>& x,
    const Vector<Real>& prm,
    const Vector<Real>& lambda,
    Vector<Real>&       out) const
{
  DenseSystemMatrix jac;
  assembleNextStateJac(step, x_next, x, prm, jac);
  jac.finalize();
  jac.applyT(lambda, out);
}

void TimeMatrixResidualEquation::applyPreviousStateJac(
    Index               step,
    const Vector<Real>& x_next,
    const Vector<Real>& x,
    const Vector<Real>& prm,
    const Vector<Real>& dir,
    Vector<Real>&       out) const
{
  DenseSystemMatrix jac;
  assemblePrevStateJac(step, x_next, x, prm, jac);
  jac.finalize();
  jac.apply(dir, out);
}

void TimeMatrixResidualEquation::applyPreviousStateJacT(
    Index               step,
    const Vector<Real>& x_next,
    const Vector<Real>& x,
    const Vector<Real>& prm,
    const Vector<Real>& lambda,
    Vector<Real>&       out) const
{
  DenseSystemMatrix jac;
  assemblePrevStateJac(step, x_next, x, prm, jac);
  jac.finalize();
  jac.applyT(lambda, out);
}

void TimeMatrixResidualEquation::applyParamJac(Index               step,
                                               const Vector<Real>& x_next,
                                               const Vector<Real>& x,
                                               const Vector<Real>& prm,
                                               const Vector<Real>& dir,
                                               Vector<Real>&       out) const
{
  DenseSystemMatrix jac;
  assembleParamJac(step, x_next, x, prm, jac);
  jac.finalize();
  jac.apply(dir, out);
}

void TimeMatrixResidualEquation::applyParamJacT(
    Index               step,
    const Vector<Real>& x_next,
    const Vector<Real>& x,
    const Vector<Real>& prm,
    const Vector<Real>& lambda,
    Vector<Real>&       out) const
{
  DenseSystemMatrix jac;
  assembleParamJac(step, x_next, x, prm, jac);
  jac.finalize();
  jac.applyT(lambda, out);
}

} // namespace problem
} // namespace femx
