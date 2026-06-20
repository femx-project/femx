#include <stdexcept>

#include <femx/problem/LeastSquaresObjective.hpp>

namespace femx
{
namespace problem
{

LeastSquaresObjective::LeastSquaresObjective(
    const problem::Observation& obs,
    const Vector<Real>&         data,
    Real                        weight)
  : obs_(obs),
    data_(data),
    weight_(weight)
{
  if (data_.size() != obs_.numObservations() || weight_ < 0.0)
  {
    throw std::runtime_error(
        "LeastSquaresObjective received inconsistent data or weight");
  }
}

Index LeastSquaresObjective::numStates() const
{
  return obs_.numStates();
}

Index LeastSquaresObjective::numParams() const
{
  return obs_.numParams();
}

Real LeastSquaresObjective::value(const Vector<Real>& state,
                                  const Vector<Real>& prm) const
{
  Vector<Real> res;
  obsResidual(state, prm, res);

  Real value_out = 0.0;
  for (Index i = 0; i < res.size(); ++i)
  {
    value_out += res[i] * res[i];
  }
  return 0.5 * weight_ * value_out;
}

void LeastSquaresObjective::stateGrad(const Vector<Real>& state,
                                      const Vector<Real>& prm,
                                      Vector<Real>&       out) const
{
  Vector<Real> wres;
  obsResidual(state, prm, wres);
  scale(wres, weight_);

  obs_.applyStateJacT(state, prm, wres, out);
  if (out.size() != numStates())
  {
    throw std::runtime_error(
        "LeastSquaresObjective state gradient size mismatch");
  }
}

void LeastSquaresObjective::paramGrad(const Vector<Real>& state,
                                      const Vector<Real>& prm,
                                      Vector<Real>&       out) const
{
  Vector<Real> wres;
  obsResidual(state, prm, wres);
  scale(wres, weight_);

  obs_.applyParamJacT(state, prm, wres, out);
  if (out.size() != numParams())
  {
    throw std::runtime_error(
        "LeastSquaresObjective parameter gradient size mismatch");
  }
}

void LeastSquaresObjective::obsResidual(const Vector<Real>& state,
                                        const Vector<Real>& prm,
                                        Vector<Real>&       out) const
{
  obs_.observe(state, prm, out);
  if (out.size() != data_.size())
  {
    throw std::runtime_error("LeastSquaresObjective observation size mismatch");
  }

  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] -= data_[i];
  }
}

void LeastSquaresObjective::scale(Vector<Real>& out,
                                  Real          factor)
{
  for (Index i = 0; i < out.size(); ++i)
  {
    out[i] *= factor;
  }
}

} // namespace problem
} // namespace femx
