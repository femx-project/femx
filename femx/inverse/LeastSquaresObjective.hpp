#pragma once

#include <stdexcept>

#include <femx/core/Types.hpp>
#include <femx/inverse/ObjectiveFunctional.hpp>
#include <femx/inverse/ObservationOperator.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Objective 0.5 * weight * ||H(u,m) - data||^2. */
class LeastSquaresObjective final : public ObjectiveFunctional
{
public:
  LeastSquaresObjective(const ObservationOperator& observation,
                        const Vector&              data,
                        real_type                  weight = 1.0)
    : observation_(observation),
      data_(data),
      weight_(weight)
  {
    if (data_.size() != observation_.numObservations() || weight_ < 0.0)
    {
      throw std::runtime_error(
          "LeastSquaresObjective received inconsistent data or weight");
    }
  }

  index_type numStates() const override
  {
    return observation_.numStates();
  }

  index_type numParams() const override
  {
    return observation_.numParams();
  }

  real_type value(const Vector& state,
                  const Vector& params) const override
  {
    Vector residual;
    observationResidual(state, params, residual);

    real_type value_out = 0.0;
    for (index_type i = 0; i < residual.size(); ++i)
    {
      value_out += residual[i] * residual[i];
    }
    return 0.5 * weight_ * value_out;
  }

  void stateGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    Vector wres;
    observationResidual(state, params, wres);
    scale(wres, weight_);

    observation_.applyStateJacT(state, params, wres, out);
    if (out.size() != numStates())
    {
      throw std::runtime_error(
          "LeastSquaresObjective state gradient size mismatch");
    }
  }

  void paramGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    Vector wres;
    observationResidual(state, params, wres);
    scale(wres, weight_);

    observation_.applyParamJacT(state, params, wres, out);
    if (out.size() != numParams())
    {
      throw std::runtime_error(
          "LeastSquaresObjective parameter gradient size mismatch");
    }
  }

private:
  void observationResidual(const Vector& state,
                           const Vector& params,
                           Vector&       out) const
  {
    observation_.observe(state, params, out);
    if (out.size() != data_.size())
    {
      throw std::runtime_error(
          "LeastSquaresObjective observation size mismatch");
    }

    for (index_type i = 0; i < out.size(); ++i)
    {
      out[i] -= data_[i];
    }
  }

  static void scale(Vector& out, real_type factor)
  {
    for (index_type i = 0; i < out.size(); ++i)
    {
      out[i] *= factor;
    }
  }

private:
  const ObservationOperator& observation_;
  Vector                     data_;
  real_type                  weight_{1.0};
};

} // namespace inverse
} // namespace femx
