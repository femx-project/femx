#pragma once

#include <femx/algebra/MatrixBuilder.hpp>
#include <femx/algebra/Vector.hpp>
#include <femx/core/Types.hpp>

namespace femx
{
namespace problem
{

struct TimeDimensions
{
  Index num_steps     = 0;
  Index num_states    = 0;
  Index num_params    = 0;
  Index num_residuals = 0;
};

struct TimeContext
{
  Index               step           = 0;
  const Vector<Real>* previous_state = nullptr;
  const Vector<Real>* next_state     = nullptr;
  const Vector<Real>* prm            = nullptr;
};

enum class VariableBlock
{
  PreviousState,
  NextState,
  Parameter
};

class TimeResidual
{
public:
  virtual ~TimeResidual() = default;

  virtual TimeDimensions dimensions() const = 0;

  virtual void residual(const TimeContext& ctx,
                        Vector<Real>&      out) const = 0;

  virtual void applyJacobian(const TimeContext&  ctx,
                             VariableBlock      wrt,
                             const Vector<Real>& dir,
                             Vector<Real>&       out) const = 0;

  virtual void applyJacobianT(const TimeContext&  ctx,
                              VariableBlock      wrt,
                              const Vector<Real>& adjoint,
                              Vector<Real>&       out) const = 0;

  virtual bool assembleJacobian(const TimeContext& ctx,
                                VariableBlock     wrt,
                                algebra::MatrixBuilder& out) const
  {
    (void) ctx;
    (void) wrt;
    (void) out;
    return false;
  }
};

} // namespace problem
} // namespace femx
