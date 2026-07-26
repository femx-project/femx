#include <array>
#include <cmath>
#include <set>
#include <string>

#include "Config.hpp"
#include "Problem.hpp"
#include "TestHelper.hpp"

#ifndef FEMX_TEST_SOURCE_DIR
#error "FEMX_TEST_SOURCE_DIR must name the femx source directory"
#endif

namespace femx::tests
{
namespace
{

std::string configPath(const char* solver,
                       const char* problem)
{
  return std::string(FEMX_TEST_SOURCE_DIR)
         + "/apps/ns-forward/configs/" + solver + "/" + problem
         + "/Config.json";
}

TestOutcome shippedConfigsUseCanonicalOptions()
{
  TestStatus status(__func__);

  constexpr std::array<const char*, 2> solvers{{"petsc", "resolve"}};
  constexpr std::array<const char*, 3> problems{
      {"cavity", "stenosis", "straighttube"}};
  for (const char* solver : solvers)
  {
    for (const char* problem : problems)
    {
      const auto prm =
          apps::ns_forward::loadConfig(configPath(solver, problem));
      status *= prm.solver.max_itrs == 5000;
      status *= std::abs(prm.solver.relative_tolerance - 1.0e-8)
                <= 1.0e-16;
      status *= !prm.mesh_file.empty();
      status *= !prm.boundary_conditions.empty();
    }
  }

  const auto straight = apps::ns_forward::loadConfig(
      configPath("resolve", "straighttube"));
  status *= straight.boundary_conditions.size() == 3;
  status *= straight.boundary_conditions[0].tag == 4;
  status *= straight.boundary_conditions[1].tag == 5;
  status *= straight.boundary_conditions[2].tag == 6;
  status *= straight.boundary_conditions[0].velocity.has_value();
  if (straight.boundary_conditions[0].velocity)
  {
    status *= straight.boundary_conditions[0].velocity->qty == "mean_velocity";
    status *= straight.boundary_conditions[0].velocity->time.size() == 5;
  }

  return status.report();
}

TestOutcome shippedCavityResolvesSharedCorners()
{
  TestStatus status(__func__);

  auto prm =
      apps::ns_forward::loadConfig(configPath("resolve", "cavity"));
  prm.time.steps = 1;
  const apps::ns_forward::Problem prob(prm);

  const std::set<Index> dofs(prob.boundary_data.dofs.begin(),
                             prob.boundary_data.dofs.end());
  status *= !dofs.empty();
  status *= static_cast<Index>(dofs.size()) == prob.boundary_data.dofs.size();

  return status.report();
}

} // namespace
} // namespace femx::tests

int main()
{
  femx::tests::TestingResults results;
  results += femx::tests::shippedConfigsUseCanonicalOptions();
  results += femx::tests::shippedCavityResolvesSharedCorners();
  return results.summary();
}
