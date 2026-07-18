#include <stdexcept>

#include <femx/inverse/SumTimeObjective.hpp>
using namespace femx::state;

namespace femx
{
namespace inverse
{

SumTimeObjective::SumTimeObjective(Index num_steps,
                                   Index num_states,
                                   Index num_param)
  : num_steps_(num_steps),
    num_states_(num_states),
    num_param_(num_param)
{
  if (num_steps_ < 0 || num_states_ < 0 || num_param_ < 0)
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

const Array<const TimeObjective*>& SumTimeObjective::terms() const noexcept
{
  return terms_;
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
  return num_param_;
}

Real SumTimeObjective::value(const TimeTrajectory& tr,
                             const HostVector&     prm) const
{
  Real value_out = 0.0;
  for (const TimeObjective* term : terms_)
  {
    value_out += term->value(tr, prm);
  }
  return value_out;
}

void SumTimeObjective::stateGrad(Index                 level,
                                 const TimeTrajectory& tr,
                                 const HostVector&     prm,
                                 HostVector&           out) const
{
  resizeOrZero(out, numStates());
  HostVector term_grad;
  for (const TimeObjective* term : terms_)
  {
    term->stateGrad(level, tr, prm, term_grad);
    addInto(term_grad, out, numStates());
  }
}

void SumTimeObjective::paramGrad(const TimeTrajectory& tr,
                                 const HostVector&     prm,
                                 HostVector&           out) const
{
  resizeOrZero(out, numParams());
  HostVector term_grad;
  for (const TimeObjective* term : terms_)
  {
    term->paramGrad(tr, prm, term_grad);
    addInto(term_grad, out, numParams());
  }
}

void SumTimeObjective::addInto(const HostVector& input,
                               HostVector&       out,
                               Index             size)
{
  checkSize(input, size);
  for (Index i = 0; i < size; ++i)
  {
    out[i] += input[i];
  }
}

void SumTimeObjective::checkSize(const HostVector& value, Index exp)
{
  if (value.size() != exp)
  {
    throw std::runtime_error("SumTimeObjective vector size mismatch");
  }
}

} // namespace inverse
} // namespace femx
