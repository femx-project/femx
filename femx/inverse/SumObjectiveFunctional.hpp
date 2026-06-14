#pragma once

#include <stdexcept>
#include <vector>

#include <femx/core/Types.hpp>
#include <femx/inverse/ObjectiveFunctional.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/** @brief Non-owning sum of objective terms with matching dimensions. */
class SumObjectiveFunctional final : public ObjectiveFunctional
{
public:
  SumObjectiveFunctional(index_type num_states,
                         index_type num_params)
    : num_states_(num_states),
      num_params_(num_params)
  {
    if (num_states_ < 0 || num_params_ < 0)
    {
      throw std::runtime_error(
          "SumObjectiveFunctional received invalid dimensions");
    }
  }

  SumObjectiveFunctional& add(const ObjectiveFunctional& term)
  {
    if (term.numStates() != numStates() || term.numParams() != numParams())
    {
      throw std::runtime_error(
          "SumObjectiveFunctional received term with inconsistent dimensions");
    }
    terms_.push_back(&term);
    return *this;
  }

  index_type numStates() const override
  {
    return num_states_;
  }

  index_type numParams() const override
  {
    return num_params_;
  }

  real_type value(const Vector& state,
                  const Vector& params) const override
  {
    real_type value_out = 0.0;
    for (const ObjectiveFunctional* term : terms_)
    {
      value_out += term->value(state, params);
    }
    return value_out;
  }

  void stateGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    resize(out, numStates());
    Vector term_grad;
    for (const ObjectiveFunctional* term : terms_)
    {
      term->stateGrad(state, params, term_grad);
      addInto(term_grad, out, numStates());
    }
  }

  void paramGrad(const Vector& state,
                 const Vector& params,
                 Vector&       out) const override
  {
    resize(out, numParams());
    Vector term_grad;
    for (const ObjectiveFunctional* term : terms_)
    {
      term->paramGrad(state, params, term_grad);
      addInto(term_grad, out, numParams());
    }
  }

private:
  static void resize(Vector& out, index_type size)
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

  static void addInto(const Vector& input, Vector& out, index_type size)
  {
    if (input.size() != size)
    {
      throw std::runtime_error(
          "SumObjectiveFunctional term gradient size mismatch");
    }
    for (index_type i = 0; i < size; ++i)
    {
      out[i] += input[i];
    }
  }

private:
  index_type                              num_states_{0};
  index_type                              num_params_{0};
  std::vector<const ObjectiveFunctional*> terms_;
};

} // namespace inverse
} // namespace femx
