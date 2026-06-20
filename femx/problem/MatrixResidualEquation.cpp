#include <femx/problem/MatrixResidualEquation.hpp>
#include <femx/algebra/backends/native/DenseSystemMatrix.hpp>

using namespace femx::algebra;

namespace femx
{
namespace problem
{

void MatrixResidualEquation::applyStateJac(const Vector<Real>& state,
                                           const Vector<Real>& prm,
                                           const Vector<Real>& dir,
                                           Vector<Real>&       out) const
{
  DenseSystemMatrix jac;
  assembleStateJac(state, prm, jac);
  jac.finalize();
  jac.apply(dir, out);
}

void MatrixResidualEquation::applyStateJacT(const Vector<Real>& state,
                                            const Vector<Real>& prm,
                                            const Vector<Real>& lambda,
                                            Vector<Real>&       out) const
{
  DenseSystemMatrix jac;
  assembleStateJac(state, prm, jac);
  jac.finalize();
  jac.applyT(lambda, out);
}

void MatrixResidualEquation::applyParamJac(const Vector<Real>& state,
                                           const Vector<Real>& prm,
                                           const Vector<Real>& dir,
                                           Vector<Real>&       out) const
{
  DenseSystemMatrix jac;
  assembleParamJac(state, prm, jac);
  jac.finalize();
  jac.apply(dir, out);
}

void MatrixResidualEquation::applyParamJacT(const Vector<Real>& state,
                                            const Vector<Real>& prm,
                                            const Vector<Real>& lambda,
                                            Vector<Real>&       out) const
{
  DenseSystemMatrix jac;
  assembleParamJac(state, prm, jac);
  jac.finalize();
  jac.applyT(lambda, out);
}

} // namespace problem
} // namespace femx
