#include <stdexcept>

#include <femx/problem/TimeObservationData.hpp>

namespace femx
{
namespace problem
{

TimeObservationData sampleTimeObs(
    const TimeObservationOperator& obs,
    const solve::TimeStateTrajectory& tr,
    const Vector<Real>&            prm)
{
  if (tr.empty())
  {
    throw std::runtime_error("sampleTimeObs received empty trajectory");
  }
  if (tr.numSteps() != obs.numSteps() || tr.numStates() != obs.numStates()
      || prm.size() != obs.numParams())
  {
    throw std::runtime_error("sampleTimeObs dimension mismatch");
  }

  TimeObservationData data(tr.numLevels(), obs.numObservations());
  for (Index level = 0; level < tr.numLevels(); ++level)
  {
    Vector<Real> level_data = data[level];
    obs.observe(level, tr[level], prm, level_data);
  }
  return data;
}

} // namespace problem
} // namespace femx
