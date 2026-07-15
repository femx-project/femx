#include <cmath>
#include <stdexcept>
#include <utility>

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
  if (num_states_ < 0 || num_param_ < 0)
  {
    throw std::runtime_error(
        "LeastSquaresObjective received invalid dimensions");
  }
}

LeastSquaresObjective::LeastSquaresObjective(
    Index        num_states,
    Index        num_param,
    Vector<Real> state_target,
    Vector<Real> state_weights,
    Vector<Real> param_target,
    Vector<Real> param_weights)
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

void LeastSquaresObjective::setStateTerm(Vector<Real> target, Real weight)
{
  setStateTerm(std::move(target), uniformWeights(num_states_, weight));
}

void LeastSquaresObjective::setStateTerm(Vector<Real> target,
                                         Vector<Real> weights)
{
  checkTerm(target, weights, num_states_, "state");
  state_target_  = std::move(target);
  state_weights_ = std::move(weights);
}

void LeastSquaresObjective::setParamTerm(Vector<Real> target, Real weight)
{
  setParamTerm(std::move(target), uniformWeights(num_param_, weight));
}

void LeastSquaresObjective::setParamTerm(Vector<Real> target,
                                         Vector<Real> weights)
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

Real LeastSquaresObjective::value(const Vector<Real>& state,
                                  const Vector<Real>& prm) const
{
  checkInputSizes(state, prm);
  return termValue(state, state_target_, state_weights_)
         + termValue(prm, param_target_, param_weights_);
}

void LeastSquaresObjective::stateGrad(const Vector<Real>& state,
                                      const Vector<Real>& prm,
                                      Vector<Real>&       out) const
{
  checkInputSizes(state, prm);
  termGrad(state, state_target_, state_weights_, out);
  if (out.empty())
  {
    out.resize(num_states_);
  }
}

void LeastSquaresObjective::paramGrad(const Vector<Real>& state,
                                      const Vector<Real>& prm,
                                      Vector<Real>&       out) const
{
  checkInputSizes(state, prm);
  termGrad(prm, param_target_, param_weights_, out);
  if (out.empty())
  {
    out.resize(num_param_);
  }
}

Vector<Real> LeastSquaresObjective::uniformWeights(Index size, Real weight)
{
  if (weight < 0.0 || !std::isfinite(weight))
  {
    throw std::runtime_error(
        "LeastSquaresObjective received invalid weight");
  }
  return Vector<Real>(size, weight);
}

Real LeastSquaresObjective::termValue(const Vector<Real>& x,
                                      const Vector<Real>& target,
                                      const Vector<Real>& weights)
{
  if (target.empty())
  {
    return 0.0;
  }

  Real value = 0.0;
  for (Index i = 0; i < x.size(); ++i)
  {
    const Real diff  = x[i] - target[i];
    value           += 0.5 * weights[i] * diff * diff;
  }
  return value;
}

void LeastSquaresObjective::termGrad(const Vector<Real>& x,
                                     const Vector<Real>& target,
                                     const Vector<Real>& weights,
                                     Vector<Real>&       out)
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
    const Vector<Real>& state,
    const Vector<Real>& prm) const
{
  if (state.size() != num_states_ || prm.size() != num_param_)
  {
    throw std::runtime_error(
        "LeastSquaresObjective received inconsistent vector sizes");
  }
}

void LeastSquaresObjective::checkTerm(const Vector<Real>& target,
                                      const Vector<Real>& weights,
                                      Index               size,
                                      const char*         name) const
{
  if (target.size() != size || weights.size() != size)
  {
    throw std::runtime_error(
        std::string("LeastSquaresObjective ") + name + " term size mismatch");
  }
  for (Index i = 0; i < weights.size(); ++i)
  {
    if (weights[i] < 0.0 || !std::isfinite(weights[i]))
    {
      throw std::runtime_error(
          std::string("LeastSquaresObjective ") + name
          + " weight must be nonnegative");
    }
  }
}

} // namespace inverse
} // namespace femx
