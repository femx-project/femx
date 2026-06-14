#pragma once

#include <femx/eq/ResidualEquation.hpp>
#include <femx/system/SystemMatrix.hpp>
#include <femx/system/native/DenseSystemMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace equation
{

/** @brief Residual equation that can assemble Jacobian matrices. */
class AssembledResidualEquation : public ResidualEquation
{
public:
  ~AssembledResidualEquation() override = default;

  virtual void assembleStateJac(const Vector& state,
                                const Vector& params,
                                system::SystemMatrix& out) const = 0;

  virtual void assembleParamJac(const Vector& state,
                                const Vector& params,
                                system::SystemMatrix& out) const = 0;

  void applyStateJac(const Vector& state,
                     const Vector& params,
                     const Vector& dir,
                     Vector&       out) const override
  {
    system::DenseSystemMatrix jacobian;
    assembleStateJac(state, params, jacobian);
    jacobian.finalize();
    jacobian.apply(dir, out);
  }

  void applyStateJacT(const Vector& state,
                      const Vector& params,
                      const Vector& lambda,
                      Vector&       out) const override
  {
    system::DenseSystemMatrix jacobian;
    assembleStateJac(state, params, jacobian);
    jacobian.finalize();
    jacobian.applyT(lambda, out);
  }

  void applyParamJac(const Vector& state,
                     const Vector& params,
                     const Vector& dir,
                     Vector&       out) const override
  {
    system::DenseSystemMatrix jacobian;
    assembleParamJac(state, params, jacobian);
    jacobian.finalize();
    jacobian.apply(dir, out);
  }

  void applyParamJacT(const Vector& state,
                      const Vector& params,
                      const Vector& lambda,
                      Vector&       out) const override
  {
    system::DenseSystemMatrix jacobian;
    assembleParamJac(state, params, jacobian);
    jacobian.finalize();
    jacobian.applyT(lambda, out);
  }
};

} // namespace equation
} // namespace femx
