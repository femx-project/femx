#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
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
                        const Vector<Real>&        data,
                        Real                       weight = 1.0)
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

  Index numStates() const override
  {
    return observation_.numStates();
  }

  Index numParams() const override
  {
    return observation_.numParams();
  }

  Real value(const Vector<Real>& state,
             const Vector<Real>& params) const override
  {
    Vector<Real> res;
    observationResidual(state, params, res);

    Real value_out = 0.0;
    for (Index i = 0; i < res.size(); ++i)
    {
      value_out += res[i] * res[i];
    }
    return 0.5 * weight_ * value_out;
  }

  void stateGrad(const Vector<Real>& state,
                 const Vector<Real>& params,
                 Vector<Real>&       out) const override
  {
    Vector<Real> wres;
    observationResidual(state, params, wres);
    scale(wres, weight_);

    observation_.applyStateJacT(state, params, wres, out);
    if (out.size() != numStates())
    {
      throw std::runtime_error(
          "LeastSquaresObjective state gradient size mismatch");
    }
  }

  void paramGrad(const Vector<Real>& state,
                 const Vector<Real>& params,
                 Vector<Real>&       out) const override
  {
    Vector<Real> wres;
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
  void observationResidual(const Vector<Real>& state,
                           const Vector<Real>& params,
                           Vector<Real>&       out) const
  {
    observation_.observe(state, params, out);
    if (out.size() != data_.size())
    {
      throw std::runtime_error(
          "LeastSquaresObjective observation size mismatch");
    }

    for (Index i = 0; i < out.size(); ++i)
    {
      out[i] -= data_[i];
    }
  }

  static void scale(Vector<Real>& out, Real factor)
  {
    for (Index i = 0; i < out.size(); ++i)
    {
      out[i] *= factor;
    }
  }

private:
  const ObservationOperator& observation_;
  Vector<Real>               data_;
  Real                       weight_{1.0};
};

} // namespace inverse
} // namespace femx
