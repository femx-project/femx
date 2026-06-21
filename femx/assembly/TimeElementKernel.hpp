#pragma once

#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/common/Types.hpp>
#include <femx/problem/TimeResidual.hpp>

namespace femx
{
namespace assembly
{

/** @brief Cell-local residual and Jacobian kernel for one time step. */
class TimeElementKernel
{
public:
  virtual ~TimeElementKernel() = default;

  virtual void res(Index               step,
                   Index               ic,
                   const Vector<Real>& prev_state,
                   const Vector<Real>& next_state,
                   const Vector<Real>& prm,
                   Vector<Real>&       out) const = 0;

  virtual void jacobian(Index                  step,
                        Index                  ic,
                        problem::VariableBlock wrt,
                        const Vector<Real>&    prev_state,
                        const Vector<Real>&    next_state,
                        const Vector<Real>&    prm,
                        DenseMatrix&           out) const = 0;
};

} // namespace assembly
} // namespace femx
