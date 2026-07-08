#pragma once

#include <femx/linalg/AssemblyMatrix.hpp>
#include <femx/linalg/LinearOperator.hpp>

namespace femx
{
namespace problem
{

/**
 * @brief Pair of Jacobian operators for a residual linearization.
 *
 * Linearization exposes state and parameter Jacobians without prescribing the
 * concrete matrix or operator implementation.
 */
class Linearization
{
public:
  virtual ~Linearization() = default;

  /** @brief Jacobian with respect to the state variable. */
  virtual const linalg::LinearOperator& stateJac() const = 0;

  /** @brief Jacobian with respect to the parameter/control variable. */
  virtual const linalg::LinearOperator& paramJac() const = 0;
};

/**
 * @brief Linearization backed by mutable assembly matrices.
 *
 * MatrixLinearization exposes concrete assembly matrices to residual
 * implementations while presenting read-only linear operators to solvers.
 */
class MatrixLinearization final : public Linearization
{
public:
  MatrixLinearization(linalg::AssemblyMatrix& J_state,
                      linalg::AssemblyMatrix& J_param)
    : J_state_(J_state),
      J_param_(J_param)
  {
  }

  linalg::AssemblyMatrix& stateMat()
  {
    return J_state_;
  }

  linalg::AssemblyMatrix& paramMat()
  {
    return J_param_;
  }

  const linalg::LinearOperator& stateJac() const override
  {
    return J_state_;
  }

  const linalg::LinearOperator& paramJac() const override
  {
    return J_param_;
  }

private:
  linalg::AssemblyMatrix& J_state_;
  linalg::AssemblyMatrix& J_param_;
};

} // namespace problem
} // namespace femx
