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
    require(inserted || std::abs(it->second - val) <= 1.0e-12,
            "conflicting Dirichlet values at dof " + std::to_string(dof)
                + ", time " + std::to_string(t));
  }
  return vals;
}

void checkDofSet(const ValueMap& ref, const ValueMap& curr, Real t)
{
  require(ref.size() == curr.size(),
          "Dirichlet constrained dofs changed at time " + std::to_string(t));
  auto ref_it  = ref.begin();
  auto curr_it = curr.begin();
  for (; ref_it != ref.end(); ++ref_it, ++curr_it)
  {
    require(ref_it->first == curr_it->first,
            "Dirichlet constrained dofs changed at time "
                + std::to_string(t));
  }
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

  out.vals.resize(nstep * out.dofs.size());
  BlockVectorView<MemorySpace::Host, Real> vals(
      out.vals.data(), nstep, out.dofs.size());
  for (Index step = 0; step < nstep; ++step)
  {
    const Real     t    = static_cast<Real>(step + 1) * dt;
    const ValueMap curr = conditionVals(
        bc_at_time(t), nstate, t);
    checkDofSet(init, curr, t);

    Index col = 0;
    for (const auto& [dof, val] : curr)
    {
      (void) dof;
      vals(step, col++) = val;
    }
  }
  return out;
}

} // namespace femx::fem
