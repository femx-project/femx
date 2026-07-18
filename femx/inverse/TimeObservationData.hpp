#pragma once

#include <string>

#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/VectorView.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace inverse
{

class TimeObservationOperator;

class TimeObservationData
{
public:
  TimeObservationData() = default;
  TimeObservationData(Index num_levels, Index num_obs);

  void resize(Index num_levels, Index num_obs);

  bool  empty() const;
  Index numTimeLevels() const;
  Index numObservations() const;

  bool hasLayout() const;
  bool hasTimeLevels() const;
  bool hasTimeValues() const;

  const std::string&   sampler() const;
  const Array<Point3>& pts() const;
  const Array<Index>&  comps() const;
  const Array<Index>&  timeLevels() const;
  const HostVector&    timeValues() const;

  Index timeLevel(Index row) const;
  Real  timeValue(Index row) const;

  void setLayout(std::string   sampler,
                 Array<Point3> pts,
                 Array<Index>  comps);

  void setTimeLevels(Array<Index> levels);
  void setTimeValues(HostVector vals);

  HostVectorView      operator[](Index level);
  HostConstVectorView operator[](Index level) const;

  void setZero();

private:
  void checkLevel(Index level) const;
  void checkLayout() const;
  void checkTimeLevels() const;
  void checkTimeValues() const;

private:
  HostVector    data_; ///< Observation values stored by time level.
  Index         num_levels_{0};
  Index         num_obs_{0};
  std::string   sampler_;     ///< Name of the sampler that produced the layout.
  Array<Point3> pts_;         ///< Observation point coordinates.
  Array<Index>  comps_;       ///< Observed component at each point.
  Array<Index>  time_levels_; ///< Source time level for each row.
  HostVector    time_vals_;   ///< Physical time value for each row.
};

TimeObservationData sampleTimeObs(const TimeObservationOperator& obs,
                                  const state::TimeTrajectory&   tr,
                                  const HostVector&              prm);

void writeTimeObsData(const std::string& path, const TimeObservationData& data);

TimeObservationData readTimeObsData(const std::string& path);

} // namespace inverse
} // namespace femx
