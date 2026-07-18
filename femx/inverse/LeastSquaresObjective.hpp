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

  LeastSquaresObjective(Index      num_states,
                        Index      num_param,
                        HostVector state_target,
                        HostVector state_weights,
                        HostVector param_target,
                        HostVector param_weights);

  Index numStates() const override;
  Index numParams() const override;

  void setStateTerm(HostVector target, Real weight = 1.0);
  void setStateTerm(HostVector target, HostVector weights);

  void setParamTerm(HostVector target, Real weight = 1.0);
  void setParamTerm(HostVector target, HostVector weights);

  void clearStateTerm();
  void clearParamTerm();

  Real value(const HostVector& state,
             const HostVector& prm) const override;

  void stateGrad(const HostVector& state,
                 const HostVector& prm,
                 HostVector&       out) const override;

  void paramGrad(const HostVector& state,
                 const HostVector& prm,
                 HostVector&       out) const override;

private:
  static HostVector uniformWeights(Index size, Real weight);
  static Real       termValue(const HostVector& x,
                              const HostVector& target,
                              const HostVector& weights);
  static void       termGrad(const HostVector& x,
                             const HostVector& target,
                             const HostVector& weights,
                             HostVector&       out);

  void checkInputSizes(const HostVector& state,
                       const HostVector& prm) const;
  void checkTerm(const HostVector& target,
                 const HostVector& weights,
                 Index             size,
                 const char*       name) const;

private:
  Index num_states_{0};
  Index num_param_{0};

  HostVector state_target_;  ///< Target state values.
  HostVector state_weights_; ///< Weights for state tracking.
  HostVector param_target_;  ///< Target parameter values.
  HostVector param_weights_; ///< Weights for parameter tracking.
};

} // namespace inverse
} // namespace femx
