#pragma once

#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/inverse/ObjectiveFunctional.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Objective 0.5 * weight * ||m - reference||^2. */
class QuadraticParameterRegularization final : public ObjectiveFunctional
{
public:
  QuadraticParameterRegularization(Index         num_states,
                                   const Vector& reference,
                                   Real          weight)
    : num_states_(num_states),
      reference_(reference),
      weight_(weight)
  {
    if (num_states_ < 0 || weight_ < 0.0)
    {
      throw std::runtime_error(
          "QuadraticParameterRegularization received invalid dimensions or weight");
    }
  }

  Index numStates() const override
  {
    return num_states_;
  }

  Index numParams() const override
  {
    return reference_.size();
  }

  Real value(const Vector& state,
             const Vector& params) const override
  {
    (void) state;
    checkParams(params);

    Real value_out = 0.0;
    for (Index i = 0; i < numParams(); ++i)
    {
      const Real diff  = params[i] - reference_[i];
      value_out       += diff * diff;
    }
    return 0.5 * weight_ * value_out;
  }

  void stateGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    (void) state;
    (void) params;
    resize(out, numStates());
  }

  void paramGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    (void) state;
    checkParams(params);
    resize(out, numParams());

    for (Index i = 0; i < numParams(); ++i)
    {
      out[i] = weight_ * (params[i] - reference_[i]);
    }
  }

private:
  void checkParams(const Vector& params) const
  {
    if (params.size() != numParams())
    {
      throw std::runtime_error(
          "QuadraticParameterRegularization parameter size mismatch");
    }
  }

  static void resize(Vector& out, Index size)
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

private:
  Index  num_states_{0};
  Vector reference_;
  Real   weight_{0.0};
};

} // namespace inverse
} // namespace femx
