#include <stdexcept>

#include <femx/problem/SumTimeObjective.hpp>

using namespace std;
using namespace femx::state;

namespace femx
{
namespace problem
{

SumTimeObjective::SumTimeObjective(Index nt,
                                   Index nst,
                                   Index nprm)
  : nt_(nt),
    nst_(nst),
    nprm_(nprm)
{
  if (nt_ < 0 || nst_ < 0 || nprm_ < 0)
  {
    throw runtime_error("SumTimeObjective received invalid dimensions");
  }
}

SumTimeObjective& SumTimeObjective::add(const TimeObjective& term)
{
  if (term.numSteps() != numSteps() || term.numStates() != numStates()
      || term.numParams() != numParams())
  {
    throw runtime_error(
        "SumTimeObjective received term with inconsistent dimensions");
  }
  terms_.push_back(&term);
  return *this;
}

Index SumTimeObjective::numSteps() const
{
  return nt_;
}

Index SumTimeObjective::numStates() const
{
  return nst_;
}

Index SumTimeObjective::numParams() const
{
  return nprm_;
}

Real SumTimeObjective::value(const TimeTrajectory& tr,
                             const Vector<Real>&   prm) const
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
                                 const Vector<Real>&   prm,
                                 Vector<Real>&         out) const
{
  resizeOrZero(out, numStates());
  Vector<Real> term_grad;
  for (const TimeObjective* term : terms_)
  {
    term->stateGrad(level, tr, prm, term_grad);
    addInto(term_grad, out, numStates());
  }
}

void SumTimeObjective::paramGrad(const TimeTrajectory& tr,
                                 const Vector<Real>&   prm,
                                 Vector<Real>&         out) const
{
  resizeOrZero(out, numParams());
  Vector<Real> term_grad;
  for (const TimeObjective* term : terms_)
  {
    term->paramGrad(tr, prm, term_grad);
    addInto(term_grad, out, numParams());
  }
}

void SumTimeObjective::addInto(const Vector<Real>& input,
                               Vector<Real>&       out,
                               Index               size)
{
  checkSize(input, size);
  for (Index i = 0; i < size; ++i)
  {
    out[i] += input[i];
  }
}

void SumTimeObjective::checkSize(const Vector<Real>& value, Index exp)
{
  if (value.size() != exp)
  {
    throw runtime_error("SumTimeObjective vector size mismatch");
  }
}

} // namespace problem
} // namespace femx
