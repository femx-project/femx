#include <stdexcept>

#include <femx/problem/SumTimeObjective.hpp>

namespace femx
{
namespace problem
{

SumTimeObjective::SumTimeObjective(Index num_steps,
                                   Index num_states,
                                   Index num_prm)
  : num_steps_(num_steps),
    num_states_(num_states),
    num_prm_(num_prm)
{
  if (num_steps_ < 0 || num_states_ < 0 || num_prm_ < 0)
  {
    throw std::runtime_error("SumTimeObjective received invalid dimensions");
  }
}

SumTimeObjective& SumTimeObjective::add(const TimeObjective& term)
{
  if (term.numSteps() != numSteps() || term.numStates() != numStates()
      || term.numParams() != numParams())
  {
    throw std::runtime_error(
        "SumTimeObjective received term with inconsistent dimensions");
  }
  terms_.push_back(&term);
  return *this;
}

Index SumTimeObjective::numSteps() const
{
  return num_steps_;
}

Index SumTimeObjective::numStates() const
{
  return num_states_;
}

Index SumTimeObjective::numParams() const
{
  return num_prm_;
}

Real SumTimeObjective::value(const solve::TimeTrajectory& tr,
                             const Vector<Real>& prm) const
{
  Real value_out = 0.0;
  for (const TimeObjective* term : terms_)
  {
    value_out += term->value(tr, prm);
  }
  return value_out;
}

void SumTimeObjective::stateGrad(Index level,
                                 const solve::TimeTrajectory& tr,
                                 const Vector<Real>& prm,
                                 Vector<Real>& out) const
{
  resize(out, numStates());
  Vector<Real> term_grad;
  for (const TimeObjective* term : terms_)
  {
    term->stateGrad(level, tr, prm, term_grad);
    addInto(term_grad, out, numStates());
  }
}

void SumTimeObjective::paramGrad(const solve::TimeTrajectory& tr,
                                 const Vector<Real>& prm,
                                 Vector<Real>& out) const
{
  resize(out, numParams());
  Vector<Real> term_grad;
  for (const TimeObjective* term : terms_)
  {
    term->paramGrad(tr, prm, term_grad);
    addInto(term_grad, out, numParams());
  }
}

void SumTimeObjective::resize(Vector<Real>& out, Index size)
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

void SumTimeObjective::addInto(const Vector<Real>& input,
                               Vector<Real>& out,
                               Index size)
{
  if (input.size() != size)
  {
    throw std::runtime_error("SumTimeObjective term gradient size mismatch");
  }
  for (Index i = 0; i < size; ++i)
  {
    out[i] += input[i];
  }
}

} // namespace problem
} // namespace femx
