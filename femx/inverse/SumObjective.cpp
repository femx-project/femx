#include <femx/common/Checks.hpp>
#include <femx/inverse/SumObjective.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/native/HostContext.hpp>

namespace femx
{
namespace inverse
{

SumObjective::SumObjective(Index num_states, Index num_param)
  : num_states_(num_states),
    num_param_(num_param)
{
  require(num_states_ >= 0 && num_param_ >= 0,
          "SumObjective received invalid dimensions");
}

SumObjective& SumObjective::add(const Objective& term)
{
  require(term.numStates() == num_states_ && term.numParams() == num_param_,
          "SumObjective received term with inconsistent dimensions");
  terms_.push_back(&term);
  return *this;
}

Index SumObjective::numStates() const
{
  return num_states_;
}

Index SumObjective::numParams() const
{
  return num_param_;
}

Real SumObjective::value(const HostVector<Real>& state,
                         const HostVector<Real>& prm) const
{
  Real val = 0.0;
  for (const Objective* term : terms_)
  {
    val += term->value(state, prm);
  }
  return val;
}

void SumObjective::stateGrad(const HostVector<Real>& state,
                             const HostVector<Real>& prm,
                             HostVector<Real>&       out) const
{
  linalg::HostContext ctx;
  auto&               vec_handler = ctx.vectorHandler();
  vec_handler.resizeOrZero(out, num_states_);

  HostVector<Real> term_grad;
  for (const Objective* term : terms_)
  {
    term->stateGrad(state, prm, term_grad);
    addInto(term_grad, out, num_states_);
  }
}

void SumObjective::paramGrad(const HostVector<Real>& state,
                             const HostVector<Real>& prm,
                             HostVector<Real>&       out) const
{
  linalg::HostContext ctx;
  auto&               vec_handler = ctx.vectorHandler();
  vec_handler.resizeOrZero(out, num_param_);

  HostVector<Real> term_grad;
  for (const Objective* term : terms_)
  {
    term->paramGrad(state, prm, term_grad);
    addInto(term_grad, out, num_param_);
  }
}

void SumObjective::addInto(const HostVector<Real>& src,
                           HostVector<Real>&       out,
                           Index                   size)
{
  require(src.size() == size && out.size() == size,
          "SumObjective vector size mismatch");
  for (Index i = 0; i < size; ++i)
  {
    out[i] += src[i];
  }
}

} // namespace inverse
} // namespace femx
