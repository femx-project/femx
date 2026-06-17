#pragma once

#include <femx/eq/ResidualEquation.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/SystemMatrix.hpp>
#include <femx/system/native/DenseSystemMatrix.hpp>

namespace femx
{
namespace eq
{

/** @brief Residual equation that can assemble Jacobian matrices. */
class AssembledResidualEquation : public ResidualEquation
{
public:
  ~AssembledResidualEquation() override = default;

  virtual void assembleStateJac(const Vector<Real>&   state,
                                const Vector<Real>&   params,
                                system::SystemMatrix& out) const = 0;

  virtual void assembleParamJac(const Vector<Real>&   state,
                                const Vector<Real>&   params,
                                system::SystemMatrix& out) const = 0;

  void applyStateJac(const Vector<Real>& state,
                     const Vector<Real>& params,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    system::DenseSystemMatrix jac;
    assembleStateJac(state, params, jac);
    jac.finalize();
    jac.apply(dir, out);
  }

  void applyStateJacT(const Vector<Real>& state,
                      const Vector<Real>& params,
                      const Vector<Real>& lambda,
                      Vector<Real>&       out) const override
  {
    system::DenseSystemMatrix jac;
    assembleStateJac(state, params, jac);
    jac.finalize();
    jac.applyT(lambda, out);
  }

  void applyParamJac(const Vector<Real>& state,
                     const Vector<Real>& params,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override
  {
    system::DenseSystemMatrix jac;
    assembleParamJac(state, params, jac);
    jac.finalize();
    jac.apply(dir, out);
  }

  void applyParamJacT(const Vector<Real>& state,
                      const Vector<Real>& params,
                      const Vector<Real>& lambda,
                      Vector<Real>&       out) const override
  {
    system::DenseSystemMatrix jac;
    assembleParamJac(state, params, jac);
    jac.finalize();
    jac.applyT(lambda, out);
  }
};

} // namespace eq
} // namespace femx
