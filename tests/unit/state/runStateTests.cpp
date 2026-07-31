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
      HostVector<Real>{1.0, 2.0, 3.0}, std::move(perturbations));

  HostVector<Real> physical;
  basis.apply(HostVector<Real>{2.0, -1.0}, physical);
  status *= physical.size() == 3;
  status *= std::abs(physical[0] - 1.0) < 1.0e-12;
  status *= std::abs(physical[1] - 0.0) < 1.0e-12;
  status *= std::abs(physical[2] - 1.0) < 1.0e-12;

  HostVector<Real> coefficients;
  basis.applyT(HostVector<Real>{4.0, -2.0, 1.0}, coefficients);
  status *= coefficients.size() == 2;
  status *= std::abs(coefficients[0] - 6.5) < 1.0e-12;
  status *= std::abs(coefficients[1] - 11.0) < 1.0e-12;

  return status.report();
}

TestOutcome trajectoryExposesContiguousDataAndLevels()
{
  TestStatus status(__func__);

  state::TimeTrajectory traj(2, 3);
  Real* const           storage  = traj.data();
  status                        *= traj.numSteps() == 2;
  status                        *= traj.numTimeLevels() == 3;
  status                        *= traj.numStates() == 3;
  status                        *= traj.size() == 9;

  traj.resize(2, 3);
  status *= traj.data() == storage;

  for (Index i = 0; i < traj.size(); ++i)
  {
    traj.data()[i] = static_cast<Real>(i + 1);
  }

  const auto second  = traj.level(1);
  status            *= second.size() == 3;
  status            *= second[0] == 4.0;
  status            *= second[1] == 5.0;
  status            *= second[2] == 6.0;

  status *= traj[1].data() == second.data();

  traj.level(2)[1]  = 42.0;
  status           *= traj.data()[7] == 42.0;

  const state::TimeTrajectory& const_traj  = traj;
  status                                  *= const_traj.data()[7] == 42.0;
  status                                  *= const_traj.level(0)[2] == 3.0;

  return status.report();
}

TestOutcome trajectoryRejectsInvalidLevels()
{
  TestStatus status(__func__);

  state::TimeTrajectory traj(1, 2);

  bool negative_threw = false;
  try
  {
    (void) traj.level(-1);
  }
  catch (const std::runtime_error&)
  {
    negative_threw = true;
  }
  status *= negative_threw;

  bool past_end_threw = false;
  try
  {
    (void) traj.level(2);
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
