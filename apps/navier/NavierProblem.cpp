#include "NavierProblem.hpp"

#include "BoundaryConditions.hpp"
#include <femx/fem/ControlMap.hpp>
#include <femx/fem/TimeDirichletData.hpp>

namespace femx::apps::navier
{
namespace
{

fem::TimeDirichletData makeBoundaryData(
    const fem::MixedFESpace&                   space,
    const HostVector<BoundaryConditionConfig>& bcs,
    Index                                      num_steps,
    Real                                       dt)
{
  return fem::makeTimeDirichletData(
      space.numDofs(),
      num_steps,
      dt,
      [&space, &bcs](Real time)
      {
        return makeDirichletBoundaryConditions(space, bcs, time);
      });
}

} // namespace

NavierProblem::NavierProblem(const Config& prm)
  : model_(prm.mesh_file,
           prm.time.steps,
           prm.time.dt,
           prm.fluid),
    boundary_data_(makeBoundaryData(model_.space(),
                                    prm.boundary_conditions,
                                    model_.numSteps(),
                                    model_.dt())),
    control_map_(fem::makeControlMap(model_.numSteps(),
                                     model_.numStates(),
                                     {},
                                     boundary_data_.dofs,
                                     boundary_data_.vals,
                                     {},
                                     0,
                                     0))
{
}

const model::navier::NavierModel&
NavierProblem::model() const noexcept
{
  return model_;
}

const fem::TimeDirichletData&
NavierProblem::boundaryData() const noexcept
{
  return boundary_data_;
}

const fem::HostControlMap&
NavierProblem::controlMap() const noexcept
{
  return control_map_;
}

const HostVector<Real>&
NavierProblem::initialState() const noexcept
{
  return boundary_data_.init_state;
}

} // namespace femx::apps::navier
