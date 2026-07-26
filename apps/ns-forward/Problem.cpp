#include "Problem.hpp"

#include "BoundaryConditions.hpp"
#include <femx/fem/ControlMap.hpp>
#include <femx/fem/TimeDirichletData.hpp>

namespace femx::apps::ns_forward
{
namespace
{

fem::TimeDirichletData makeBoundaryData(
    const fem::MixedFESpace&                   space,
    const HostVector<BoundaryConditionConfig>& conditions,
    Index                                      num_steps,
    Real                                       dt)
{
  return fem::makeTimeDirichletData(
      space.numDofs(),
      num_steps,
      dt,
      [&space, &conditions](Real time)
      {
        return makeDirichletBoundaryConditions(space, conditions, time);
      });
}

} // namespace

Problem::Problem(const Config& config)
  : model(config.mesh_file,
          config.time.steps,
          config.time.dt,
          config.fluid),
    boundary_data(makeBoundaryData(model.space(),
                                   config.boundary_conditions,
                                   model.numSteps(),
                                   model.dt())),
    residual(model.residual(),
             fem::makeControlMap(model.numSteps(),
                                 model.numStates(),
                                 {},
                                 boundary_data.dofs,
                                 boundary_data.vals,
                                 {},
                                 0,
                                 0)),
    initial_state(boundary_data.init_state),
    parameters(0)
{
}

} // namespace femx::apps::ns_forward
