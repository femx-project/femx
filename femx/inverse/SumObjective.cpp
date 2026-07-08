#include <stdexcept>

#include <femx/inverse/SumObjective.hpp>

namespace femx
{
namespace inverse
{

SumObjective::SumObjective(Index num_states, Index num_params)
  : num_states_(num_states),
    num_params_(num_params)
{
  if (num_states_ < 0 || num_params_ < 0)
  {
    throw std::runtime_error("SumObjective received invalid dimensions");
  }
}

SumObjective& SumObjective::add(const Objective& term)
{
  if (term.numStates() != num_states_ || term.numParams() != num_params_)
  {
    throw std::runtime_error(
        "SumObjective received term with inconsistent dimensions");
  }
  terms_.push_back(&term);
  return *this;
}

Index SumObjective::numStates() const
{
  return num_states_;
}

Index SumObjective::numParams() const
{
  return num_params_;
}

Real SumObjective::value(const Vector<Real>& state,
                         const Vector<Real>& prm) const
{
  Real value_out = 0.0;
  for (const Objective* term : terms_)
  {
    value_out += term->value(state, prm);
  }
  return value_out;
}

void SumObjective::stateGrad(const Vector<Real>& state,
                             const Vector<Real>& prm,
                             Vector<Real>&       out) const
{
  resizeOrZero(out, num_states_);

  Vector<Real> term_grad;
  for (const Objective* term : terms_)
  {
    term->stateGrad(state, prm, term_grad);
    addInto(term_grad, out, num_states_);
  }
}

void SumObjective::paramGrad(const Vector<Real>& state,
                             const Vector<Real>& prm,
                             Vector<Real>&       out) const
{
  resizeOrZero(out, num_params_);

  Vector<Real> term_grad;
  for (const Objective* term : terms_)
  {
    term->paramGrad(state, prm, term_grad);
    addInto(term_grad, out, num_params_);
  }
}

void SumObjective::addInto(const Vector<Real>& input,
                           Vector<Real>&       out,
                           Index               size)
{
  if (input.size() != size || out.size() != size)
  {
    throw std::runtime_error("SumObjective vector size mismatch");
  }
  for (Index i = 0; i < size; ++i)
  {
    out[i] += input[i];
  }
}

} // namespace inverse
} // namespace femx
