#include <femx/eq/MatrixResidualEquation.hpp>
#include <femx/system/native/DenseSystemMatrix.hpp>

using namespace femx::system;

namespace femx
{
namespace eq
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

} // namespace eq
} // namespace femx
