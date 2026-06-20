#include <stdexcept>

#include <femx/problem/SumObjectiveFunctional.hpp>

namespace femx
{
namespace problem
{

SumObjectiveFunctional::SumObjectiveFunctional(Index num_states,
                                               Index num_prm)
  : num_states_(num_states),
    num_prm_(num_prm)
{
  if (num_states_ < 0 || num_prm_ < 0)
  {
    throw std::runtime_error(
        "SumObjectiveFunctional received invalid dimensions");
  }
}

SumObjectiveFunctional& SumObjectiveFunctional::add(
    const ObjectiveFunctional& term)
{
  if (term.numStates() != numStates() || term.numParams() != numParams())
  {
    throw std::runtime_error(
        "SumObjectiveFunctional received term with inconsistent dimensions");
  }
  terms_.push_back(&term);
  return *this;
}

Index SumObjectiveFunctional::numStates() const
{
  return num_states_;
}

Index SumObjectiveFunctional::numParams() const
{
  return num_prm_;
}

Real SumObjectiveFunctional::value(const Vector<Real>& state,
                                   const Vector<Real>& prm) const
{
  Real value_out = 0.0;
  for (const ObjectiveFunctional* term : terms_)
  {
    value_out += term->value(state, prm);
  }
  return value_out;
}

void SumObjectiveFunctional::stateGrad(const Vector<Real>& state,
                                       const Vector<Real>& prm,
                                       Vector<Real>&       out) const
{
  resize(out, numStates());
  Vector<Real> term_grad;
  for (const ObjectiveFunctional* term : terms_)
  {
    term->stateGrad(state, prm, term_grad);
    addInto(term_grad, out, numStates());
  }
}

void SumObjectiveFunctional::paramGrad(const Vector<Real>& state,
                                       const Vector<Real>& prm,
                                       Vector<Real>&       out) const
{
  resize(out, numParams());
  Vector<Real> term_grad;
  for (const ObjectiveFunctional* term : terms_)
  {
    term->paramGrad(state, prm, term_grad);
    addInto(term_grad, out, numParams());
  }
}

void SumObjectiveFunctional::resize(Vector<Real>& out,
                                    Index         size)
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

void SumObjectiveFunctional::addInto(const Vector<Real>& input,
                                     Vector<Real>&       out,
                                     Index               size)
{
  if (input.size() != size)
  {
    throw std::runtime_error(
        "SumObjectiveFunctional term gradient size mismatch");
  }
  for (Index i = 0; i < size; ++i)
  {
    out[i] += input[i];
  }
}

} // namespace problem
} // namespace femx
