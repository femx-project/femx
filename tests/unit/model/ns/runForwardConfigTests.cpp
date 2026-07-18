#include <array>
#include <cmath>
#include <set>
#include <string>

#include "TestHelper.hpp"
#include <femx/model/ns/ForwardConfig.hpp>
#include <femx/model/ns/ForwardProblem.hpp>

#ifndef FEMX_TEST_SOURCE_DIR
#error "FEMX_TEST_SOURCE_DIR must name the femx source directory"
#endif

namespace femx::tests
{
namespace
{

std::string configPath(const char* backend,
                       const char* problem)
{
  return std::string(FEMX_TEST_SOURCE_DIR)
         + "/apps/ns-forward/configs/" + backend + "/" + problem
         + "/Config.json";
}

TestOutcome shippedConfigsUseCanonicalOptions()
{
  TestStatus status(__func__);

  constexpr std::array<const char*, 2> backends{{"petsc", "resolve"}};
  constexpr std::array<const char*, 3> problems{
      {"cavity", "stenosis", "straighttube"}};
  for (const char* backend : backends)
  {
    for (const char* problem : problems)
    {
      const auto prm  = model::ns::loadConfig(configPath(backend, problem));
      status         *= prm.solver.max_itrs == 5000;
      status         *= std::abs(prm.solver.relative_tolerance - 1.0e-8)
                <= 1.0e-16;
      status *= !prm.mesh_file.empty();
      status *= !prm.bcs.empty();
    }
  }

  const auto straight = model::ns::loadConfig(
      configPath("resolve", "straighttube"));
  status *= straight.bcs.size() == 3;
  status *= straight.bcs[0].tag == 4;
  status *= straight.bcs[1].tag == 5;
  status *= straight.bcs[2].tag == 6;
  status *= straight.bcs[0].velocity.has_value();
  if (straight.bcs[0].velocity)
  {
    status *= straight.bcs[0].velocity->qty == "mean_velocity";
    status *= straight.bcs[0].velocity->time.size() == 5;
  }

  return status.report();
}

TestOutcome shippedCavityResolvesSharedCorners()
{
  TestStatus status(__func__);

  auto prm       = model::ns::loadConfig(configPath("resolve", "cavity"));
  prm.time.steps = 1;
  const model::ns::ForwardProblem prob(prm);

  const std::set<Index> dofs(prob.fixed.dofs.begin(),
                             prob.fixed.dofs.end());
  status *= !dofs.empty();
  status *= static_cast<Index>(dofs.size()) == prob.fixed.dofs.size();

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
