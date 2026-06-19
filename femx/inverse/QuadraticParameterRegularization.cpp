#include <stdexcept>

#include <femx/inverse/QuadraticParameterRegularization.hpp>

namespace femx
{
namespace inverse
{

QuadraticParameterRegularization::QuadraticParameterRegularization(
    Index               num_states,
    const Vector<Real>& reference,
    Real                weight)
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

Index QuadraticParameterRegularization::numStates() const
{
  return num_states_;
}

Index QuadraticParameterRegularization::numParams() const
{
  return reference_.size();
}

Real QuadraticParameterRegularization::value(const Vector<Real>& state,
                                             const Vector<Real>& prm) const
{
  (void) state;
  checkParams(prm);

  Real value_out = 0.0;
  for (Index i = 0; i < numParams(); ++i)
  {
    const Real diff  = prm[i] - reference_[i];
    value_out       += diff * diff;
  }
  return 0.5 * weight_ * value_out;
}

void QuadraticParameterRegularization::stateGrad(
    const Vector<Real>& state,
    const Vector<Real>& prm,
    Vector<Real>&       out) const
{
  (void) state;
  (void) prm;
  resize(out, numStates());
}

void QuadraticParameterRegularization::paramGrad(
    const Vector<Real>& state,
    const Vector<Real>& prm,
    Vector<Real>&       out) const
{
  (void) state;
  checkParams(prm);
  resize(out, numParams());

  for (Index i = 0; i < numParams(); ++i)
  {
    out[i] = weight_ * (prm[i] - reference_[i]);
  }
}

void QuadraticParameterRegularization::checkParams(
    const Vector<Real>& prm) const
{
  if (prm.size() != numParams())
  {
    throw std::runtime_error(
        "QuadraticParameterRegularization parameter size mismatch");
  }
}

void QuadraticParameterRegularization::resize(Vector<Real>& out,
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

} // namespace inverse
} // namespace femx
