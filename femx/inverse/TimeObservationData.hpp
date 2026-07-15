#pragma once

#include <string>

#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/VectorView.hpp>

namespace femx
{
namespace state
{
class TimeTrajectory;
} // namespace state

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
  Index numLevels() const;
  Index numObservations() const;

  bool hasLayout() const;
  bool hasTimeLevels() const;
  bool hasTimeValues() const;

  const std::string&    sampler() const;
  const Vector<Point3>& pts() const;
  const Vector<Index>&  comps() const;
  const Vector<Index>&  timeLevels() const;
  const Vector<Real>&   timeValues() const;

  Index timeLevel(Index row) const;
  Real  timeValue(Index row) const;

  void setLayout(std::string    sampler,
                 Vector<Point3> pts,
                 Vector<Index>  comps);

  void setTimeLevels(Vector<Index> levels);
  void setTimeValues(Vector<Real> vals);

  VectorView<Real>       operator[](Index level);
  VectorView<const Real> operator[](Index level) const;

  void setZero();

private:
  void checkLevel(Index level) const;
  void checkLayout() const;
  void checkTimeLevels() const;
  void checkTimeValues() const;

private:
  Vector<Real>   data_; ///< Observation values stored by time level.
  Index          num_levels_{0};
  Index          num_obs_{0};
  std::string    sampler_;     ///< Name of the sampler that produced the layout.
  Vector<Point3> pts_;         ///< Observation point coordinates.
  Vector<Index>  comps_;       ///< Observed component at each point.
  Vector<Index>  time_levels_; ///< Source time level for each row.
  Vector<Real>   time_values_; ///< Physical time value for each row.
};

TimeObservationData sampleTimeObs(const TimeObservationOperator& obs,
                                  const state::TimeTrajectory&   tr,
                                  const Vector<Real>&            prm);

void writeTimeObsData(const std::string& path, const TimeObservationData& data);

TimeObservationData readTimeObsData(const std::string& path);

} // namespace inverse
} // namespace femx
