#pragma once

#include <femx/linalg/LinearOperator.hpp>
#include <femx/linalg/MatrixOperator.hpp>

namespace femx
{
namespace state
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
  MatrixLinearization(linalg::MatrixOperator& J_state,
                      linalg::MatrixOperator& J_param)
    : J_state_(J_state),
      J_param_(J_param)
  {
  }

  linalg::MatrixOperator& stateMat()
  {
    return J_state_;
  }

  linalg::MatrixOperator& paramMat()
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
  linalg::MatrixOperator& J_state_;
  linalg::MatrixOperator& J_param_;
};

} // namespace state
} // namespace femx
