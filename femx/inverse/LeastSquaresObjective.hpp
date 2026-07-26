#pragma once

#include <femx/common/Types.hpp>
#include <femx/inverse/Objective.hpp>
#include <femx/linalg/Vector.hpp>

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
  LeastSquaresObjective(Index num_states, Index num_param);

  LeastSquaresObjective(Index            num_states,
                        Index            num_param,
                        HostVector<Real> state_target,
                        HostVector<Real> state_weights,
                        HostVector<Real> param_target,
                        HostVector<Real> param_weights);

  Index numStates() const override;
  Index numParams() const override;

  void setStateTerm(HostVector<Real> target, Real weight = 1.0);
  void setStateTerm(HostVector<Real> target, HostVector<Real> weights);

  void setParamTerm(HostVector<Real> target, Real weight = 1.0);
  void setParamTerm(HostVector<Real> target, HostVector<Real> weights);

  void clearStateTerm();
  void clearParamTerm();

  Real value(const HostVector<Real>& state,
             const HostVector<Real>& prm) const override;

  void stateGrad(const HostVector<Real>& state,
                 const HostVector<Real>& prm,
                 HostVector<Real>&       out) const override;

  void paramGrad(const HostVector<Real>& state,
                 const HostVector<Real>& prm,
                 HostVector<Real>&       out) const override;

private:
  static HostVector<Real> uniformWeights(Index size, Real weight);
  static Real             termValue(const HostVector<Real>& x,
                                    const HostVector<Real>& target,
                                    const HostVector<Real>& weights);
  static void             termGrad(const HostVector<Real>& x,
                                   const HostVector<Real>& target,
                                   const HostVector<Real>& weights,
                                   HostVector<Real>&       out);

  void checkInputSizes(const HostVector<Real>& state,
                       const HostVector<Real>& prm) const;
  void checkTerm(const HostVector<Real>& target,
                 const HostVector<Real>& weights,
                 Index                   size,
                 const char*             name) const;

private:
  Index num_states_{0};
  Index num_param_{0};

  HostVector<Real> state_target_;  ///< Target state values.
  HostVector<Real> state_weights_; ///< Weights for state tracking.
  HostVector<Real> param_target_;  ///< Target parameter values.
  HostVector<Real> param_weights_; ///< Weights for parameter tracking.
};

} // namespace inverse
} // namespace femx
