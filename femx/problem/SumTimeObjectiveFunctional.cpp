#include <stdexcept>

#include <femx/problem/SumTimeObjectiveFunctional.hpp>

using namespace femx::solve;

namespace femx
{
namespace problem
{

SumTimeObjectiveFunctional::SumTimeObjectiveFunctional(Index num_steps,
                                                       Index num_states,
                                                       Index num_prm)
  : num_steps_(num_steps),
    num_states_(num_states),
    num_prm_(num_prm)
{
  if (num_steps_ < 0 || num_states_ < 0 || num_prm_ < 0)
  {
    throw std::runtime_error(
        "SumTimeObjectiveFunctional received invalid dimensions");
  }
}

SumTimeObjectiveFunctional& SumTimeObjectiveFunctional::add(
    const TimeObjectiveFunctional& term)
{
  if (term.numSteps() != numSteps() || term.numStates() != numStates()
      || term.numParams() != numParams())
  {
    throw std::runtime_error(
        "SumTimeObjectiveFunctional received term with inconsistent dimensions");
  }
  terms_.push_back(&term);
  return *this;
}

Index SumTimeObjectiveFunctional::numSteps() const
{
  return num_steps_;
}

Index SumTimeObjectiveFunctional::numStates() const
{
  return num_states_;
}

Index SumTimeObjectiveFunctional::numParams() const
{
  return num_prm_;
}

Real SumTimeObjectiveFunctional::value(const TimeStateTrajectory& tr,
                                       const Vector<Real>&        prm) const
{
  Real value_out = 0.0;
  for (const TimeObjectiveFunctional* term : terms_)
  {
    value_out += term->value(tr, prm);
  }
  return value_out;
}

void SumTimeObjectiveFunctional::stateGrad(
    Index                      level,
    const TimeStateTrajectory& tr,
    const Vector<Real>&        prm,
    Vector<Real>&              out) const
{
  resize(out, numStates());
  Vector<Real> term_grad;
  for (const TimeObjectiveFunctional* term : terms_)
  {
    term->stateGrad(level, tr, prm, term_grad);
    addInto(term_grad, out, numStates());
  }
}

void SumTimeObjectiveFunctional::paramGrad(const TimeStateTrajectory& tr,
                                           const Vector<Real>&        prm,
                                           Vector<Real>&              out) const
{
  resize(out, numParams());
  Vector<Real> term_grad;
  for (const TimeObjectiveFunctional* term : terms_)
  {
    term->paramGrad(tr, prm, term_grad);
    addInto(term_grad, out, numParams());
  }
}

void SumTimeObjectiveFunctional::resize(Vector<Real>& out,
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

void SumTimeObjectiveFunctional::addInto(const Vector<Real>& input,
                                         Vector<Real>&       out,
                                         Index               size)
{
  if (input.size() != size)
  {
    throw std::runtime_error(
        "SumTimeObjectiveFunctional term gradient size mismatch");
  }
  for (Index i = 0; i < size; ++i)
  {
    out[i] += input[i];
  }
}

} // namespace problem
} // namespace femx
