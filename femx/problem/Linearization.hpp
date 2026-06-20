#pragma once

#include <femx/algebra/LinearOperator.hpp>
#include <femx/algebra/MatrixOperator.hpp>

namespace femx
{
namespace problem
{

class Linearization
{
public:
  virtual ~Linearization() = default;

  virtual const algebra::LinearOperator& stateJacobian() const = 0;
  virtual const algebra::LinearOperator& paramJacobian() const = 0;
};

class MatrixLinearization final : public Linearization
{
public:
  MatrixLinearization(algebra::MatrixOperator& state_jac,
                      algebra::MatrixOperator& param_jac)
    : state_jac_(state_jac),
      param_jac_(param_jac)
  {
  }

  algebra::MatrixOperator& stateMatrix()
  {
    return state_jac_;
  }

  algebra::MatrixOperator& paramMatrix()
  {
    return param_jac_;
  }

  const algebra::LinearOperator& stateJacobian() const override
  {
    return state_jac_;
  }

  const algebra::LinearOperator& paramJacobian() const override
  {
    return param_jac_;
  }

private:
  algebra::MatrixOperator& state_jac_;
  algebra::MatrixOperator& param_jac_;
};

} // namespace problem
} // namespace femx
