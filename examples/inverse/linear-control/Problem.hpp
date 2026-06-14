#pragma once

#include <iostream>

#include <femx/eq/AssembledResidualEquation.hpp>
#include <femx/inverse/LeastSquaresObjective.hpp>
#include <femx/inverse/ObservationOperator.hpp>
#include <femx/inverse/QuadraticParameterRegularization.hpp>
#include <femx/inverse/SumObjectiveFunctional.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/system/SystemMatrix.hpp>

namespace femx
{
namespace examples_inverse_linear_control
{

inline void resize(Vector& out, Index size)
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

class LinearResidualEquation final : public eq::AssembledResidualEquation
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

  Index numRes() const override
  {
    return 2;
  }

  void res(const Vector& state,
           const Vector& params,
           Vector&       out) const override
  {
    resize(out, numRes());
    out[0] = 2.0 * state[0] + 3.0 * state[1]
             + 5.0 * params[0] - 2.0 * params[1];
    out[1] = 7.0 * state[0] + 11.0 * state[1]
             + 13.0 * params[0] + 4.0 * params[1];
  }

  void assembleStateJac(const Vector&         state,
                        const Vector&         params,
                        system::SystemMatrix& out) const override
  {
    (void) state;
    (void) params;
    out.resize(numRes(), numStates());
    out.setZero();
    out.set(0, 0, 2.0);
    out.set(0, 1, 3.0);
    out.set(1, 0, 7.0);
    out.set(1, 1, 11.0);
  }

  void assembleParamJac(const Vector&         state,
                        const Vector&         params,
                        system::SystemMatrix& out) const override
  {
    (void) state;
    (void) params;
    out.resize(numRes(), numParams());
    out.setZero();
    out.set(0, 0, 5.0);
    out.set(0, 1, -2.0);
    out.set(1, 0, 13.0);
    out.set(1, 1, 4.0);
  }
};

class StateTrackingObservation final : public inverse::ObservationOperator
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

  Index numObservations() const override
  {
    return 2;
  }

  void observe(const Vector& state,
               const Vector& params,
               Vector&       out) const override
  {
    (void) params;
    resize(out, numObservations());
    out[0] = state[0];
    out[1] = state[1];
  }

  void applyStateJac(const Vector& state,
                     const Vector& params,
                     const Vector& dir,
                     Vector&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numObservations());
    out[0] = dir[0];
    out[1] = dir[1];
  }

  void applyStateJacT(const Vector& state,
                      const Vector& params,
                      const Vector& dir,
                      Vector&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numStates());
    out[0] = dir[0];
    out[1] = dir[1];
  }

  void applyParamJac(const Vector& state,
                     const Vector& params,
                     const Vector& dir,
                     Vector&       out) const override
  {
    (void) state;
    (void) params;
    (void) dir;
    resize(out, numObservations());
  }

  void applyParamJacT(const Vector& state,
                      const Vector& params,
                      const Vector& dir,
                      Vector&       out) const override
  {
    (void) state;
    (void) params;
    (void) dir;
    resize(out, numParams());
  }
};

inline Vector makeStateTarget()
{
  Vector target(2);
  target[0] = 0.25;
  target[1] = -0.75;
  return target;
}

inline Vector makeZeroParams()
{
  Vector reference(2);
  reference[0] = 0.0;
  reference[1] = 0.0;
  return reference;
}

struct LinearControlObjectiveParts
{
  StateTrackingObservation                  observation;
  Vector                                    target;
  inverse::LeastSquaresObjective            tracking;
  Vector                                    reference;
  inverse::QuadraticParameterRegularization regularization;
  inverse::SumObjectiveFunctional           objective;

  LinearControlObjectiveParts()
    : target(makeStateTarget()),
      tracking(observation, target),
      reference(makeZeroParams()),
      regularization(2, reference, 0.25),
      objective(2, 2)
  {
    objective.add(tracking).add(regularization);
  }
};

inline void printVector(const char*   name,
                        const Vector& x,
                        std::ostream& out = std::cout)
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

} // namespace examples_inverse_linear_control
} // namespace femx
