#include <femx/common/Checks.hpp>
#include <femx/inverse/SumTimeObjective.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/native/HostContext.hpp>
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
  require(num_steps_ >= 0 && num_states_ >= 0 && num_param_ >= 0,
          "SumTimeObjective received invalid dimensions");
}

SumTimeObjective& SumTimeObjective::add(const TimeObjective& term)
{
  require(term.numSteps() == numSteps() && term.numStates() == numStates()
              && term.numParams() == numParams(),
          "SumTimeObjective received term with inconsistent dimensions");
  terms_.push_back(&term);
  return *this;
}

const HostVector<const TimeObjective*>& SumTimeObjective::terms() const noexcept
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

Real SumTimeObjective::value(const TimeTrajectory&   traj,
                             const HostVector<Real>& prm) const
{
  Real val = 0.0;
  for (const TimeObjective* term : terms_)
  {
    val += term->value(traj, prm);
  }
  return val;
}

void SumTimeObjective::stateGrad(Index                   level,
                                 const TimeTrajectory&   traj,
                                 const HostVector<Real>& prm,
                                 HostVector<Real>&       out) const
{
  linalg::HostContext ctx;
  auto&               vec_handler = ctx.vectorHandler();
  vec_handler.resizeOrZero(out, numStates());
  HostVector<Real> term_grad;
  for (const TimeObjective* term : terms_)
  {
    term->stateGrad(level, traj, prm, term_grad);
    addInto(term_grad, out, numStates());
  }
}

void SumTimeObjective::paramGrad(const TimeTrajectory&   traj,
                                 const HostVector<Real>& prm,
                                 HostVector<Real>&       out) const
{
  linalg::HostContext ctx;
  auto&               vec_handler = ctx.vectorHandler();
  vec_handler.resizeOrZero(out, numParams());
  HostVector<Real> term_grad;
  for (const TimeObjective* term : terms_)
  {
    term->paramGrad(traj, prm, term_grad);
    addInto(term_grad, out, numParams());
  }
}

void SumTimeObjective::addInto(const HostVector<Real>& src,
                               HostVector<Real>&       out,
                               Index                   size)
{
  checkSize(src, size);
  for (Index i = 0; i < size; ++i)
  {
    out[i] += src[i];
  }
}

void SumTimeObjective::checkSize(const HostVector<Real>& val, Index exp)
{
  require(val.size() == exp,
          "SumTimeObjective vector size mismatch");
}

} // namespace inverse
} // namespace femx
