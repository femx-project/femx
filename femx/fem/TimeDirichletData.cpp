#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

#include <femx/common/Checks.hpp>
#include <femx/fem/TimeDirichletData.hpp>
#include <femx/linalg/View.hpp>

namespace femx::fem
{
namespace
{

using ValueMap = std::map<Index, Real>;

ValueMap conditionVals(const DirichletBC& bc,
                       Index              nstate,
                       Real               t)
{
  require(bc.dofs().size() == bc.vals().size(),
          "DirichletBC has inconsistent data");

  ValueMap vals;
  for (Index i = 0; i < bc.dofs().size(); ++i)
  {
    const Index dof = bc.dofs()[i];
    const Real  val = bc.vals()[i];
    require(dof >= 0 && dof < nstate,
            "Dirichlet dof is out of range");
    require(std::isfinite(val), "Dirichlet value must be finite");

    const auto [it, inserted] = vals.emplace(dof, val);
    if (!inserted && std::abs(it->second - val) > 1.0e-12)
    {
      throw std::runtime_error(
          "conflicting Dirichlet values at dof " + std::to_string(dof)
          + ", time " + std::to_string(t));
    }
  }
  return vals;
}

[[noreturn]] void throwChangedDofSet(Real t)
{
  throw std::runtime_error(
      "Dirichlet constrained dofs changed at time " + std::to_string(t));
}

} // namespace

TimeDirichletData makeTimeDirichletData(
    Index                    nstate,
    Index                    nstep,
    Real                     dt,
    const DirichletBCAtTime& bc_at_time)
{
  require(nstate > 0 && nstep > 0 && std::isfinite(dt) && dt > 0.0,
          "makeTimeDirichletData received invalid dimensions");
  require(static_cast<bool>(bc_at_time),
          "makeTimeDirichletData requires a boundary-condition callback");

  const ValueMap init = conditionVals(bc_at_time(0.0), nstate, 0.0);

  TimeDirichletData out;
  out.init_state.resize(nstate);
  for (const auto& [dof, val] : init)
  {
    out.dofs.push_back(dof);
    out.init_state[dof] = val;
  }

  HostVector<Index> col_by_dof(nstate, -1);
  for (Index col = 0; col < out.dofs.size(); ++col)
  {
    col_by_dof[out.dofs[col]] = col;
  }
  HostVector<Index> seen(out.dofs.size(), -1);

  out.vals.resize(nstep * out.dofs.size());
  BlockVectorView<MemorySpace::Host, Real> vals(
      out.vals.data(), nstep, out.dofs.size());
  for (Index step = 0; step < nstep; ++step)
  {
    const Real        t    = static_cast<Real>(step + 1) * dt;
    const DirichletBC curr = bc_at_time(t);
    require(curr.dofs().size() == curr.vals().size(),
            "DirichletBC has inconsistent data");

    Index unique_dofs = 0;
    for (Index i = 0; i < curr.dofs().size(); ++i)
    {
      const Index dof = curr.dofs()[i];
      const Real  val = curr.vals()[i];
      require(dof >= 0 && dof < nstate,
              "Dirichlet dof is out of range");
      require(std::isfinite(val), "Dirichlet value must be finite");

      const Index col = col_by_dof[dof];
      if (col < 0)
      {
        throwChangedDofSet(t);
      }
      if (seen[col] == step)
      {
        if (std::abs(vals(step, col) - val) > 1.0e-12)
        {
          throw std::runtime_error(
              "conflicting Dirichlet values at dof "
              + std::to_string(dof) + ", time " + std::to_string(t));
        }
        continue;
      }

      seen[col]       = step;
      vals(step, col) = val;
      ++unique_dofs;
    }
    if (unique_dofs != out.dofs.size())
    {
      throwChangedDofSet(t);
    }
  }
  return out;
}

} // namespace femx::fem
