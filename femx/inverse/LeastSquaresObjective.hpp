#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/inverse/Objective.hpp>

namespace femx
{
namespace inverse
{

/**
 * @brief Diagonal least-squares objective for stationary problems.
 *
 * Represents weighted state and parameter tracking terms,
 * 0.5 ||u - u_d||_W^2 + 0.5 ||m - m_d||_M^2.
 */
class LeastSquaresObjective final : public Objective
{
public:
  LeastSquaresObjective(Index num_states, Index num_params);

  LeastSquaresObjective(Index        num_states,
                        Index        num_params,
                        Vector<Real> state_target,
                        Vector<Real> state_weights,
                        Vector<Real> param_target,
                        Vector<Real> param_weights);

  Index numStates() const override;
  Index numParams() const override;

  void setStateTerm(Vector<Real> target, Real weight = 1.0);
  void setStateTerm(Vector<Real> target, Vector<Real> weights);

  void setParamTerm(Vector<Real> target, Real weight = 1.0);
  void setParamTerm(Vector<Real> target, Vector<Real> weights);

  void clearStateTerm();
  void clearParamTerm();

  Real value(const Vector<Real>& state,
             const Vector<Real>& prm) const override;

  void stateGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override;

  void paramGrad(const Vector<Real>& state,
                 const Vector<Real>& prm,
                 Vector<Real>&       out) const override;

private:
  static Vector<Real> uniformWeights(Index size, Real weight);
  static Real         termValue(const Vector<Real>& x,
                                const Vector<Real>& target,
                                const Vector<Real>& weights);
  static void         termGrad(const Vector<Real>& x,
                               const Vector<Real>& target,
                               const Vector<Real>& weights,
                               Vector<Real>&       out);

  void checkInputSizes(const Vector<Real>& state,
                       const Vector<Real>& prm) const;
  void checkTerm(const Vector<Real>& target,
                 const Vector<Real>& weights,
                 Index               size,
                 const char*         name) const;

private:
  Index num_states_{0};
  Index num_params_{0};

  Vector<Real> state_target_;
  Vector<Real> state_weights_;
  Vector<Real> param_target_;
  Vector<Real> param_weights_;
};

} // namespace inverse
} // namespace femx
