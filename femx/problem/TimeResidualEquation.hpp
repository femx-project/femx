#pragma once

#include <femx/core/Types.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx
{
namespace problem
{

/**
 * @brief Residual for one time step, R_k(u_{k+1}, u_k, m) = 0.
 *
 * Step indices satisfy 0 <= step < numSteps(). The "next" state is
 * u_{step+1}; the "previous" state is u_step.
 */
class TimeResidualEquation
{
public:
  virtual ~TimeResidualEquation() = default;

  virtual Index numSteps() const  = 0;
  virtual Index numStates() const = 0;
  virtual Index numParams() const = 0;
  virtual Index numRes() const    = 0;

  virtual void res(Index               step,
                   const Vector<Real>& x_next,
                   const Vector<Real>& x,
                   const Vector<Real>& prm,
                   Vector<Real>&       out) const = 0;

  virtual void applyNextStateJac(Index               step,
                                 const Vector<Real>& x_next,
                                 const Vector<Real>& x,
                                 const Vector<Real>& prm,
                                 const Vector<Real>& dir,
                                 Vector<Real>&       out) const = 0;

  virtual void applyNextStateJacT(Index               step,
                                  const Vector<Real>& x_next,
                                  const Vector<Real>& x,
                                  const Vector<Real>& prm,
                                  const Vector<Real>& lambda,
                                  Vector<Real>&       out) const = 0;

  virtual void applyPreviousStateJac(Index               step,
                                     const Vector<Real>& x_next,
                                     const Vector<Real>& x,
                                     const Vector<Real>& prm,
                                     const Vector<Real>& dir,
                                     Vector<Real>&       out) const = 0;

  virtual void applyPreviousStateJacT(Index               step,
                                      const Vector<Real>& x_next,
                                      const Vector<Real>& x,
                                      const Vector<Real>& prm,
                                      const Vector<Real>& lambda,
                                      Vector<Real>&       out) const = 0;

  virtual void applyParamJac(Index               step,
                             const Vector<Real>& x_next,
                             const Vector<Real>& x,
                             const Vector<Real>& prm,
                             const Vector<Real>& dir,
                             Vector<Real>&       out) const = 0;

  virtual void applyParamJacT(Index               step,
                              const Vector<Real>& x_next,
                              const Vector<Real>& x,
                              const Vector<Real>& prm,
                              const Vector<Real>& lambda,
                              Vector<Real>&       out) const = 0;
};

} // namespace problem
} // namespace femx
