#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

#include <femx/fem/TimeDirichletData.hpp>
#include <femx/linalg/BlockVectorView.hpp>

namespace femx::fem
{
namespace
{

using ValueMap = std::map<Index, Real>;

ValueMap conditionValues(const DirichletBC& condition,
                         Index              num_states,
                         Real               time)
{
  if (condition.dofs().size() != condition.values().size())
  {
    throw std::runtime_error("DirichletBC has inconsistent data");
  }

  ValueMap values;
  for (Index i = 0; i < condition.dofs().size(); ++i)
  {
    const Index dof   = condition.dofs()[i];
    const Real  value = condition.values()[i];
    if (dof < 0 || dof >= num_states)
    {
      throw std::runtime_error("Dirichlet dof is out of range");
    }
    if (!std::isfinite(value))
    {
      throw std::runtime_error("Dirichlet value must be finite");
    }

    const auto [it, inserted] = values.emplace(dof, value);
    if (!inserted && std::abs(it->second - value) > 1.0e-12)
    {
      throw std::runtime_error(
          "conflicting Dirichlet values at dof " + std::to_string(dof)
          + ", time " + std::to_string(time));
    }
  }
  return values;
}

void checkDofSet(const ValueMap& expected,
                 const ValueMap& actual,
                 Real            time)
{
  if (expected.size() != actual.size())
  {
    throw std::runtime_error(
        "Dirichlet constrained dofs changed at time " + std::to_string(time));
  }
  auto expected_it = expected.begin();
  auto actual_it   = actual.begin();
  for (; expected_it != expected.end(); ++expected_it, ++actual_it)
  {
    if (expected_it->first != actual_it->first)
    {
      throw std::runtime_error(
          "Dirichlet constrained dofs changed at time "
          + std::to_string(time));
    }
  }
}

} // namespace

TimeDirichletData makeTimeDirichletData(
    Index                    num_states,
    Index                    steps,
    Real                     dt,
    const DirichletBCAtTime& bc_at_time)
{
  if (num_states <= 0 || steps <= 0 || !std::isfinite(dt) || dt <= 0.0)
  {
    throw std::runtime_error(
        "makeTimeDirichletData received invalid dimensions");
  }
  if (!bc_at_time)
  {
    throw std::runtime_error(
        "makeTimeDirichletData requires a boundary-condition callback");
  }

  const ValueMap initial = conditionValues(
      bc_at_time(0.0), num_states, 0.0);

  TimeDirichletData out;
  out.initial_state.resize(num_states);
  out.initial_state.setZero();
  for (const auto& [dof, value] : initial)
  {
    out.dofs.push_back(dof);
    out.initial_state[dof] = value;
  }

  out.values.resize(steps * out.dofs.size());
  BlockVectorView<Real> values(out.values.data(), steps, out.dofs.size());
  for (Index step = 0; step < steps; ++step)
  {
    const Real     time    = static_cast<Real>(step + 1) * dt;
    const ValueMap current = conditionValues(
        bc_at_time(time), num_states, time);
    checkDofSet(initial, current, time);

    Index column = 0;
    for (const auto& [dof, value] : current)
    {
      (void) dof;
      values(step, column++) = value;
    }
  }
  return out;
}

} // namespace femx::fem
