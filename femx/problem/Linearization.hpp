#pragma once

#include <femx/linalg/LinearOperator.hpp>
#include <femx/linalg/MatrixOperator.hpp>

namespace femx
{
namespace problem
{

class Linearization
{
public:
  virtual ~Linearization() = default;

  virtual const linalg::LinearOperator& stateJacobian() const = 0;
  virtual const linalg::LinearOperator& paramJacobian() const = 0;
};

class MatrixLinearization final : public Linearization
{
public:
  MatrixLinearization(linalg::MatrixOperator& state_jac,
                      linalg::MatrixOperator& param_jac)
    : state_jac_(state_jac),
      param_jac_(param_jac)
  {
  }

  linalg::MatrixOperator& stateMatrix()
  {
    return state_jac_;
  }

  linalg::MatrixOperator& paramMatrix()
  {
    return param_jac_;
  }

  const linalg::LinearOperator& stateJacobian() const override
  {
    return state_jac_;
  }

  const linalg::LinearOperator& paramJacobian() const override
  {
    return param_jac_;
  }

private:
  linalg::MatrixOperator& state_jac_;
  linalg::MatrixOperator& param_jac_;
};

} // namespace problem
} // namespace femx
