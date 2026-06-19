#pragma once

#include <string>
#include <vector>

#include <femx/common/Math.hpp>
#include <femx/eq/TimeStateTrajectory.hpp>
#include <femx/inverse/TimeObservationOperator.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

class TimeObservationData
{
public:
  TimeObservationData() = default;

  TimeObservationData(Index num_levels,
                      Index num_observations);

  void resize(Index num_levels,
              Index num_observations);

  bool empty() const;

  Index numLevels() const;

  Index numObservations() const;

  Index size() const;

  bool hasLayout() const;

  bool hasTimeLevels() const;

  bool hasTimeValues() const;

  const std::string& sampler() const;

  const std::vector<Point3>& points() const;

  const Vector<Index>& components() const;

  const Vector<Index>& timeLevels() const;

  const Vector<Real>& timeValues() const;

  Index timeLevel(Index row) const;

  Real timeValue(Index row) const;

  void setLayout(std::string         sampler,
                 std::vector<Point3> points,
                 Vector<Index>       components);

  void setTimeLevels(Vector<Index> levels);

  void setTimeValues(Vector<Real> values);

  Vector<Real> operator[](Index level);

  Vector<Real> operator[](Index level) const;

  void setZero();

private:
  void checkLevel(Index level) const;
  void checkLayout() const;
  void checkTimeLevels() const;
  void checkTimeValues() const;

private:
  Vector<Real>        data_;
  Index               num_levels_{0};
  Index               num_obs_{0};
  std::string         sampler_;
  std::vector<Point3> points_;
  Vector<Index>       components_;
  Vector<Index>       time_levels_;
  Vector<Real>        time_values_;
};

/** @brief Sample a reference trajectory through a time observation operator. */
TimeObservationData sampleTimeObs(
    const TimeObservationOperator& obs,
    const eq::TimeStateTrajectory& tr,
    const Vector<Real>&            prm);

void writeTimeObsData(const std::string&         path,
                      const TimeObservationData& data);

TimeObservationData readTimeObsData(const std::string& path);

} // namespace inverse
} // namespace femx
