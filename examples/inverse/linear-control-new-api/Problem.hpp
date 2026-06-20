#pragma once

#include <iostream>
#include <stdexcept>

#include <femx/algebra/Vector.hpp>
#include <femx/problem/Objective.hpp>
#include <femx/problem/Residual.hpp>
#include <femx/solve/Newton.hpp>
#include <femx/solve/ReducedFunctional.hpp>
#include <femx/algebra/DenseLinearSolver.hpp>
#include <femx/algebra/backends/native/DenseSystemMatrix.hpp>

namespace femx
{
namespace examples_inverse_linear_control_new_api
{

inline void resize(Vector<Real>& out, Index size)
{
  if (out.size() != size)
  {
    out.resize(size);
  }
  else
  {
    out.setZero();
  }
}

class LinearControlProblem final : public problem::Residual
{
public:
  problem::Dimensions dimensions() const override
  {
    return {2, 2, 2};
  }

  void residual(const Vector<Real>& state,
                const Vector<Real>& prm,
                Vector<Real>&       out) const override
  {
    resize(out, 2);
    out[0] = 2.0 * state[0] + 3.0 * state[1]
             + 5.0 * prm[0] - 2.0 * prm[1];
    out[1] = 7.0 * state[0] + 11.0 * state[1]
             + 13.0 * prm[0] + 4.0 * prm[1];
  }

  void linearize(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 problem::Linearization& out) const override
  {
    (void) state;
    (void) prm;

    auto* matrix_out = dynamic_cast<problem::MatrixLinearization*>(&out);
    if (matrix_out == nullptr)
    {
      throw std::runtime_error(
          "LinearControlProblem requires MatrixLinearization");
    }

    algebra::MatrixOperator& state_jac = matrix_out->stateMatrix();
    state_jac.resize(2, 2);
    state_jac.setZero();
    state_jac.set(0, 0, 2.0);
    state_jac.set(0, 1, 3.0);
    state_jac.set(1, 0, 7.0);
    state_jac.set(1, 1, 11.0);
    state_jac.finalize();

    algebra::MatrixOperator& param_jac = matrix_out->paramMatrix();
    param_jac.resize(2, 2);
    param_jac.setZero();
    param_jac.set(0, 0, 5.0);
    param_jac.set(0, 1, -2.0);
    param_jac.set(1, 0, 13.0);
    param_jac.set(1, 1, 4.0);
    param_jac.finalize();
  }
};

class LinearControlObjective final : public problem::Objective
{
public:
  Index numStates() const override
  {
    return 2;
  }

  Index numParams() const override
  {
    return 2;
  }

  Real value(const Vector<Real>& state,
             const Vector<Real>& prm) const override
  {
    const Real d0 = state[0] - target0();
    const Real d1 = state[1] - target1();
    return 0.5 * (d0 * d0 + d1 * d1)
           + 0.5 * regularizationWeight() * (prm[0] * prm[0] + prm[1] * prm[1]);
  }

  void stateGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    (void) prm;
    resize(out, numStates());
    out[0] = state[0] - target0();
    out[1] = state[1] - target1();
  }

  void paramGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    (void) state;
    resize(out, numParams());
    out[0] = regularizationWeight() * prm[0];
    out[1] = regularizationWeight() * prm[1];
  }

  static Real target0()
  {
    return 0.25;
  }

  static Real target1()
  {
    return -0.75;
  }

  static Real regularizationWeight()
  {
    return 0.25;
  }
};

struct LinearControlSetup
{
  LinearControlProblem        problem;
  LinearControlObjective      objective;
  algebra::DenseSystemMatrix  state_jac;
  algebra::DenseSystemMatrix  param_jac;
  problem::MatrixLinearization linearization;
  algebra::DenseLinearSolver   linear_solver;
  solve::Newton                state_solver;
  solve::ReducedFunctional     functional;

  LinearControlSetup()
    : linearization(state_jac, param_jac),
      state_solver(problem, linearization, linear_solver),
      functional(problem, objective, state_solver, linear_solver)
  {
    state_solver.options().residual_tolerance = 1.0e-12;
  }
};

inline Vector<Real> makeInitialParams()
{
  Vector<Real> prm(2);
  prm[0] = 0.05;
  prm[1] = -0.02;
  return prm;
}

inline void printVector(const char*         name,
                        const Vector<Real>& x,
                        std::ostream&       out = std::cout)
{
  out << name << " = [";
  for (Index i = 0; i < x.size(); ++i)
  {
    if (i != 0)
    {
      out << ", ";
    }
    out << x[i];
  }
  out << "]\n";
}

} // namespace examples_inverse_linear_control_new_api
} // namespace femx
