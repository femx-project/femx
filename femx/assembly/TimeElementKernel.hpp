#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/TimeResidual.hpp>

namespace femx
{
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

  virtual void res(Index                    step,
                   Index                    ie,
                   problem::TimeHistoryView hist,
                   const Vector<Real>&      nxt,
                   const Vector<Real>&      prm,
                   Vector<Real>&            out) const = 0;

  virtual void jacobian(Index                    step,
                        Index                    ie,
                        problem::VariableBlock   wrt,
                        problem::TimeHistoryView hist,
                        const Vector<Real>&      nxt,
                        const Vector<Real>&      prm,
                        DenseMatrix&             out) const = 0;
};

} // namespace assembly
} // namespace femx
