#include <cmath>
#include <stdexcept>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/inverse/LeastSquaresObjective.hpp>

namespace femx
{
namespace inverse
{

LeastSquaresObjective::LeastSquaresObjective(Index num_states,
                                             Index num_param)
  : num_states_(num_states),
    num_param_(num_param)
{
  require(num_states_ >= 0 && num_param_ >= 0,
          "LeastSquaresObjective received invalid dimensions");
}

LeastSquaresObjective::LeastSquaresObjective(
    Index      num_states,
    Index      num_param,
    HostVector state_target,
    HostVector state_weights,
    HostVector param_target,
    HostVector param_weights)
  : LeastSquaresObjective(num_states, num_param)
{
  setStateTerm(std::move(state_target), std::move(state_weights));
  setParamTerm(std::move(param_target), std::move(param_weights));
}

Index LeastSquaresObjective::numStates() const
{
  return num_states_;
}

Index LeastSquaresObjective::numParams() const
{
  return num_param_;
}

void LeastSquaresObjective::setStateTerm(HostVector target, Real weight)
{
  setStateTerm(std::move(target), uniformWeights(num_states_, weight));
}

void LeastSquaresObjective::setStateTerm(HostVector target,
                                         HostVector weights)
{
  checkTerm(target, weights, num_states_, "state");
  state_target_  = std::move(target);
  state_weights_ = std::move(weights);
}

void LeastSquaresObjective::setParamTerm(HostVector target, Real weight)
{
  setParamTerm(std::move(target), uniformWeights(num_param_, weight));
}

void LeastSquaresObjective::setParamTerm(HostVector target,
                                         HostVector weights)
{
  checkTerm(target, weights, num_param_, "parameter");
  param_target_  = std::move(target);
  param_weights_ = std::move(weights);
}

void LeastSquaresObjective::clearStateTerm()
{
  state_target_.clear();
  state_weights_.clear();
}

void LeastSquaresObjective::clearParamTerm()
{
  param_target_.clear();
  param_weights_.clear();
}

Real LeastSquaresObjective::value(const HostVector& state,
                                  const HostVector& prm) const
{
  checkInputSizes(state, prm);
  return termValue(state, state_target_, state_weights_)
         + termValue(prm, param_target_, param_weights_);
}

void LeastSquaresObjective::stateGrad(const HostVector& state,
                                      const HostVector& prm,
                                      HostVector&       out) const
{
  checkInputSizes(state, prm);
  termGrad(state, state_target_, state_weights_, out);
  if (out.empty())
  {
    out.resize(num_states_);
  }
}

void LeastSquaresObjective::paramGrad(const HostVector& state,
                                      const HostVector& prm,
                                      HostVector&       out) const
{
  checkInputSizes(state, prm);
  termGrad(prm, param_target_, param_weights_, out);
  if (out.empty())
  {
    out.resize(num_param_);
  }
}

HostVector LeastSquaresObjective::uniformWeights(Index size, Real weight)
{
  require(weight >= 0.0 && std::isfinite(weight),
          "LeastSquaresObjective received invalid weight");
  return HostVector(size, weight);
}

Real LeastSquaresObjective::termValue(const HostVector& x,
                                      const HostVector& target,
                                      const HostVector& weights)
{
  if (target.empty())
  {
    return 0.0;
  }

  Real val = 0.0;
  for (Index i = 0; i < x.size(); ++i)
  {
    const Real diff  = x[i] - target[i];
    val             += 0.5 * weights[i] * diff * diff;
  }
  return val;
}

void LeastSquaresObjective::termGrad(const HostVector& x,
                                     const HostVector& target,
                                     const HostVector& weights,
                                     HostVector&       out)
{
  resizeOrZero(out, x.size());
  if (target.empty())
  {
    return;
  }

  for (Index i = 0; i < x.size(); ++i)
  {
    out[i] = weights[i] * (x[i] - target[i]);
  }
}

void LeastSquaresObjective::checkInputSizes(
    const HostVector& state,
    const HostVector& prm) const
{
  require(state.size() == num_states_ && prm.size() == num_param_,
          "LeastSquaresObjective received inconsistent vector sizes");
}

void LeastSquaresObjective::checkTerm(const HostVector& target,
                                      const HostVector& weights,
                                      Index             size,
                                      const char*       name) const
{
  require(target.size() == size && weights.size() == size,
          std::string("LeastSquaresObjective ") + name
              + " term size mismatch");
  for (Index i = 0; i < weights.size(); ++i)
  {
    require(weights[i] >= 0.0 && std::isfinite(weights[i]),
            std::string("LeastSquaresObjective ") + name
                + " weight must be nonnegative");
  }
}

} // namespace inverse
} // namespace femx
