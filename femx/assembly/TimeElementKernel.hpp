#pragma once

#include <femx/common/Types.hpp>
#include <femx/state/TimeResidual.hpp>

namespace femx
{
class DenseMatrix;

template <typename T>
class Vector;

namespace assembly
{

/**
 * @brief Elem-local residual and Jacobian kernel for one time step.
 *
 * Implementations operate on local history, next-state, and parameter values
 * gathered for a single element.
 */
class TimeElementKernel
{
public:
  virtual ~TimeElementKernel() = default;

  virtual void res(Index                  step,
                   Index                  ie,
                   state::TimeHistoryView hist,
                   const Vector<Real>&    nxt,
                   const Vector<Real>&    prm,
                   Vector<Real>&          out) const = 0;

  virtual void jacobian(Index                  step,
                        Index                  ie,
                        state::VariableBlock   wrt,
                        state::TimeHistoryView hist,
                        const Vector<Real>&    nxt,
                        const Vector<Real>&    prm,
                        DenseMatrix&           out) const = 0;
};

} // namespace assembly
} // namespace femx
