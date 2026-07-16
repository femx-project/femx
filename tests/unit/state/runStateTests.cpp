#include <stdexcept>

#include "TestHelper.hpp"
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace tests
{
namespace
{

TestOutcome trajectoryExposesContiguousDataAndLevels()
{
  TestStatus status(__func__);

  state::TimeTrajectory trajectory(2, 3);
  status *= trajectory.numSteps() == 2;
  status *= trajectory.numTimeLevels() == 3;
  status *= trajectory.numStates() == 3;
  status *= trajectory.size() == 9;

  for (Index i = 0; i < trajectory.size(); ++i)
  {
    trajectory.data()[i] = static_cast<Real>(i + 1);
  }

  const auto second  = trajectory.level(1);
  status            *= second.size() == 3;
  status            *= second[0] == 4.0;
  status            *= second[1] == 5.0;
  status            *= second[2] == 6.0;

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
  results += femx::tests::trajectoryExposesContiguousDataAndLevels();
  results += femx::tests::trajectoryRejectsInvalidLevels();
  return results.summary();
}
