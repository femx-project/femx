#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/MatrixBuilder.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace problem
{

struct TimeDims
{
  Index num_steps     = 0;
  Index num_states    = 0;
  Index num_params    = 0;
  Index num_residuals = 0;
};

struct TimeContext
{
  Index               step       = 0;
  const Vector<Real>* prev_state = nullptr;
  const Vector<Real>* next_state = nullptr;
  const Vector<Real>* prm        = nullptr;
};

enum class VariableBlock
{
  PrevState,
  NextState,
  Param
};

class TimeResidual
{
public:
  virtual ~TimeResidual() = default;

  virtual TimeDims dimensions() const = 0;

  virtual void residual(const TimeContext& ctx,
                        Vector<Real>&      out) const = 0;

  virtual void applyJac(const TimeContext&  ctx,
                        VariableBlock       wrt,
                        const Vector<Real>& dir,
                        Vector<Real>&       out) const = 0;

  virtual void applyJacT(const TimeContext&  ctx,
                         VariableBlock       wrt,
                         const Vector<Real>& adjoint,
                         Vector<Real>&       out) const = 0;

  virtual bool assembleJacobian(const TimeContext&     ctx,
                                VariableBlock          wrt,
                                linalg::MatrixBuilder& out) const
  {
    (void) ctx;
    (void) wrt;
    (void) out;
    return false;
  }
};

} // namespace problem
} // namespace femx
