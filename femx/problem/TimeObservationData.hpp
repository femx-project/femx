#pragma once

#include <string>

#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/problem/TimeObservationOperator.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace problem
{

class TimeObservationData
{
public:
  TimeObservationData() = default;

  TimeObservationData(Index nl, Index num_observations);

  void resize(Index nl, Index num_observations);

  bool  empty() const;
  Index numLevels() const;
  Index numObservations() const;
  Index size() const;

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

  Vector<Real> operator[](Index level);
  Vector<Real> operator[](Index level) const;

  void setZero();

private:
  void checkLevel(Index level) const;
  void checkLayout() const;
  void checkTimeLevels() const;
  void checkTimeValues() const;

private:
  Vector<Real>   data_;
  Index          nl_{0};
  Index          num_obs_{0};
  std::string    sampler_;
  Vector<Point3> pts_;
  Vector<Index>  comps_;
  Vector<Index>  time_levels_;
  Vector<Real>   time_values_;
};

TimeObservationData sampleTimeObs(const TimeObservationOperator& obs,
                                  const state::TimeTrajectory&   tr,
                                  const Vector<Real>&            prm);

void writeTimeObsData(const std::string& path, const TimeObservationData& data);

TimeObservationData readTimeObsData(const std::string& path);

} // namespace problem
} // namespace femx
