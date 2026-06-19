#pragma once

#include <femx/eq/ResidualEquation.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/SystemMatrix.hpp>

namespace femx
{
namespace eq
{

/** @brief Residual equation that can assemble Jacobian matrices. */
class MatrixResidualEquation : public ResidualEquation
{
public:
  ~MatrixResidualEquation() override = default;

  virtual void assembleStateJac(const Vector<Real>&   state,
                                const Vector<Real>&   prm,
                                system::SystemMatrix& out) const = 0;

  virtual void assembleParamJac(const Vector<Real>&   state,
                                const Vector<Real>&   prm,
                                system::SystemMatrix& out) const = 0;

  void applyStateJac(const Vector<Real>& state,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override;

  void applyStateJacT(const Vector<Real>& state,
                      const Vector<Real>& prm,
                      const Vector<Real>& lambda,
                      Vector<Real>&       out) const override;

  void applyParamJac(const Vector<Real>& state,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override;

  void applyParamJacT(const Vector<Real>& state,
                      const Vector<Real>& prm,
                      const Vector<Real>& lambda,
                      Vector<Real>&       out) const override;
};

} // namespace eq
} // namespace femx
