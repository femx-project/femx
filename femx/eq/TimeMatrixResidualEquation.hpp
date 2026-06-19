#pragma once

#include <femx/eq/TimeResidualEquation.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/SystemMatrix.hpp>

namespace femx
{
namespace eq
{

/** @brief Time residual equation that can assemble step Jacobian matrices. */
class TimeMatrixResidualEquation : public TimeResidualEquation
{
public:
  ~TimeMatrixResidualEquation() override = default;

  virtual void assembleNextStateJac(Index                 step,
                                    const Vector<Real>&   x_next,
                                    const Vector<Real>&   x,
                                    const Vector<Real>&   prm,
                                    system::SystemMatrix& out) const = 0;

  virtual void assemblePrevStateJac(Index                 step,
                                        const Vector<Real>&   x_next,
                                        const Vector<Real>&   x,
                                        const Vector<Real>&   prm,
                                        system::SystemMatrix& out) const = 0;

  virtual void assembleParamJac(Index                 step,
                                const Vector<Real>&   x_next,
                                const Vector<Real>&   x,
                                const Vector<Real>&   prm,
                                system::SystemMatrix& out) const = 0;

  void applyNextStateJac(Index               step,
                         const Vector<Real>& x_next,
                         const Vector<Real>& x,
                         const Vector<Real>& prm,
                         const Vector<Real>& dir,
                         Vector<Real>&       out) const override;

  void applyNextStateJacT(Index               step,
                          const Vector<Real>& x_next,
                          const Vector<Real>& x,
                          const Vector<Real>& prm,
                          const Vector<Real>& lambda,
                          Vector<Real>&       out) const override;

  void applyPreviousStateJac(Index               step,
                             const Vector<Real>& x_next,
                             const Vector<Real>& x,
                             const Vector<Real>& prm,
                             const Vector<Real>& dir,
                             Vector<Real>&       out) const override;

  void applyPreviousStateJacT(Index               step,
                              const Vector<Real>& x_next,
                              const Vector<Real>& x,
                              const Vector<Real>& prm,
                              const Vector<Real>& lambda,
                              Vector<Real>&       out) const override;

  void applyParamJac(Index               step,
                     const Vector<Real>& x_next,
                     const Vector<Real>& x,
                     const Vector<Real>& prm,
                     const Vector<Real>& dir,
                     Vector<Real>&       out) const override;

  void applyParamJacT(Index               step,
                      const Vector<Real>& x_next,
                      const Vector<Real>& x,
                      const Vector<Real>& prm,
                      const Vector<Real>& lambda,
                      Vector<Real>&       out) const override;
};

} // namespace eq
} // namespace femx
