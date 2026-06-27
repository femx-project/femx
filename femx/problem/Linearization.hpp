#pragma once

#include <femx/linalg/operator/LinearOperator.hpp>
#include <femx/linalg/operator/MatrixOperator.hpp>

namespace femx
{
namespace problem
{

class Linearization
{
public:
  virtual ~Linearization() = default;

  virtual const linalg::LinearOperator& stateJac() const = 0;
  virtual const linalg::LinearOperator& paramJac() const = 0;
};

class MatrixLinearization final : public Linearization
{
public:
  MatrixLinearization(linalg::MatrixOperator& J_state,
                      linalg::MatrixOperator& J_param)
    : J_state_(J_state),
      J_param_(J_param)
  {
  }

  linalg::MatrixOperator& stateMatrix()
  {
    return J_state_;
  }

  linalg::MatrixOperator& paramMatrix()
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

} // namespace problem
} // namespace femx
