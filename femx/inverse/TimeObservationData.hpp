#pragma once

#include <string>

#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/View.hpp>
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

  const std::string&        sampler() const;
  const HostVector<Point3>& pts() const;
  const HostVector<Index>&  comps() const;
  const HostVector<Index>&  timeLevels() const;
  const HostVector<Real>&   timeValues() const;

  Index timeLevel(Index row) const;
  Real  timeValue(Index row) const;

  void setLayout(std::string        sampler,
                 HostVector<Point3> pts,
                 HostVector<Index>  comps);

  void setTimeLevels(HostVector<Index> levels);
  void setTimeValues(HostVector<Real> vals);

  HostVectorView<Real>       operator[](Index level);
  HostVectorView<const Real> operator[](Index level) const;

  void setZero();

private:
  void checkLevel(Index level) const;
  void checkLayout() const;
  void checkTimeLevels() const;
  void checkTimeValues() const;

private:
  HostVector<Real>   data_; ///< Observation values stored by time level.
  Index              num_levels_{0};
  Index              num_obs_{0};
  std::string        sampler_;     ///< Name of the sampler that produced the layout.
  HostVector<Point3> pts_;         ///< Observation point coordinates.
  HostVector<Index>  comps_;       ///< Observed component at each point.
  HostVector<Index>  time_levels_; ///< Source time level for each row.
  HostVector<Real>   time_vals_;   ///< Physical time value for each row.
};

TimeObservationData sampleTimeObs(const TimeObservationOperator& obs,
                                  const state::TimeTrajectory&   tr,
                                  const HostVector<Real>&        prm);

void writeTimeObsData(const std::string& path, const TimeObservationData& data);

TimeObservationData readTimeObsData(const std::string& path);

} // namespace inverse
} // namespace femx
