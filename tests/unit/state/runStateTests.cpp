#include <stdexcept>
#include <utility>

#include "TestHelper.hpp"
#include <femx/state/EnsembleBasis.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace tests
{
namespace
{

TestOutcome ensembleBasisUsesDenseProducts()
{
  TestStatus status(__func__);

  DenseMatrix perturbations(3, 2);
  perturbations(0, 0) = 1.0;
  perturbations(0, 1) = 2.0;
  perturbations(1, 0) = -1.0;
  perturbations(1, 1) = 0.0;
  perturbations(2, 0) = 0.5;
  perturbations(2, 1) = 3.0;

  const state::EnsembleBasis basis(
      HostVector{1.0, 2.0, 3.0}, std::move(perturbations));

  HostVector physical;
  basis.apply(HostVector{2.0, -1.0}, physical);
  status *= physical.size() == 3;
  status *= std::abs(physical[0] - 1.0) < 1.0e-12;
  status *= std::abs(physical[1] - 0.0) < 1.0e-12;
  status *= std::abs(physical[2] - 1.0) < 1.0e-12;

  HostVector coefficients;
  basis.applyT(HostVector{4.0, -2.0, 1.0}, coefficients);
  status *= coefficients.size() == 2;
  status *= std::abs(coefficients[0] - 6.5) < 1.0e-12;
  status *= std::abs(coefficients[1] - 11.0) < 1.0e-12;

  return status.report();
}

TestOutcome trajectoryExposesContiguousDataAndLevels()
{
  TestStatus status(__func__);

  state::TimeTrajectory trajectory(2, 3);
  Real* const           storage  = trajectory.data();
  status                        *= trajectory.numSteps() == 2;
  status                        *= trajectory.numTimeLevels() == 3;
  status                        *= trajectory.numStates() == 3;
  status                        *= trajectory.size() == 9;

  trajectory.resize(2, 3);
  status *= trajectory.data() == storage;

  for (Index i = 0; i < trajectory.size(); ++i)
  {
    trajectory.data()[i] = static_cast<Real>(i + 1);
  }

  const auto second  = trajectory.level(1);
  status            *= second.size() == 3;
  status            *= second[0] == 4.0;
  status            *= second[1] == 5.0;
  status            *= second[2] == 6.0;

  const auto view  = trajectory.view();
  status          *= view.numTimeLevels() == 3;
  status          *= view[1].data() == second.data();

  trajectory.level(2)[1]  = 42.0;
  status                 *= trajectory.data()[7] == 42.0;

  const state::TimeTrajectory& const_trajectory  = trajectory;
  status                                        *= const_trajectory.data()[7] == 42.0;
  status                                        *= const_trajectory.level(0)[2] == 3.0;

  return status.report();
}

TestOutcome trajectoryRejectsInvalidLevels()
{
  TestStatus status(__func__);

  state::TimeTrajectory trajectory(1, 2);

  bool negative_threw = false;
  try
  {
    (void) trajectory.level(-1);
  }
  catch (const std::runtime_error&)
  {
    negative_threw = true;
  }
  status *= negative_threw;

  bool past_end_threw = false;
  try
  {
    (void) trajectory.level(2);
  }
  catch (const std::runtime_error&)
  {
    past_end_threw = true;
  }
  status *= past_end_threw;

  return status.report();
}

} // namespace
} // namespace tests
} // namespace femx

int main()
{
  femx::tests::TestingResults results;
  results += femx::tests::ensembleBasisUsesDenseProducts();
  results += femx::tests::trajectoryExposesContiguousDataAndLevels();
  results += femx::tests::trajectoryRejectsInvalidLevels();
  return results.summary();
}
