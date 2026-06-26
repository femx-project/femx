#pragma once

#include <iostream>
#include <stdexcept>

#include <femx/linalg/DenseLinearSolver.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/native/DenseMatrixOperator.hpp>
#include <femx/problem/Objective.hpp>
#include <femx/problem/Residual.hpp>
#include <femx/state/NewtonStateSolver.hpp>
#include <femx/state/ReducedFunctional.hpp>

namespace femx
{
namespace examples_inverse_linear_ctr_new_api
{

class LinearControlProblem final : public problem::Residual
{
public:
  problem::Dimensions dims() const override
  {
    return {2, 2, 2};
  }

  void res(const Vector<Real>& state,
           const Vector<Real>& prm,
           Vector<Real>&       out) const override
  {
    resizeOrZero(out, 2);
    out[0] = 2.0 * state[0] + 3.0 * state[1]
             + 5.0 * prm[0] - 2.0 * prm[1];
    out[1] = 7.0 * state[0] + 11.0 * state[1]
             + 13.0 * prm[0] + 4.0 * prm[1];
  }

  void linearize(const Vector<Real>&     state,
                 const Vector<Real>&     prm,
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

    linalg::MatrixOperator& J_state = matrix_out->stateMatrix();
    J_state.resize(2, 2);
    J_state.setZero();
    J_state.set(0, 0, 2.0);
    J_state.set(0, 1, 3.0);
    J_state.set(1, 0, 7.0);
    J_state.set(1, 1, 11.0);
    J_state.finalize();

    linalg::MatrixOperator& J_param = matrix_out->paramMatrix();
    J_param.resize(2, 2);
    J_param.setZero();
    J_param.set(0, 0, 5.0);
    J_param.set(0, 1, -2.0);
    J_param.set(1, 0, 13.0);
    J_param.set(1, 1, 4.0);
    J_param.finalize();
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
    resizeOrZero(out, numStates());
    out[0] = state[0] - target0();
    out[1] = state[1] - target1();
  }

  void paramGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override
  {
    (void) state;
    resizeOrZero(out, numParams());
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
  LinearControlProblem         problem;
  LinearControlObjective       obj;
  linalg::DenseMatrixOperator  J_state;
  linalg::DenseMatrixOperator  J_param;
  problem::MatrixLinearization lin;
  linalg::DenseLinearSolver    lin_solver;
  state::NewtonStateSolver     state_solver;
  state::ReducedFunctional     fn;

  LinearControlSetup()
    : lin(J_state, J_param),
      state_solver(problem, lin, lin_solver),
      fn(problem, obj, state_solver, lin, lin_solver)
  {
    state_solver.opts().residual_tolerance = 1.0e-12;
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

} // namespace examples_inverse_linear_ctr_new_api
} // namespace femx
