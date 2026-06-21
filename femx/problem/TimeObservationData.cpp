#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/problem/TimeObservationData.hpp>

namespace femx
{
namespace problem
{

TimeObservationData::TimeObservationData(Index num_levels,
                                         Index num_observations)
{
  resize(num_levels, num_observations);
}

void TimeObservationData::resize(Index num_levels, Index num_observations)
{
  if (num_levels < 0 || num_observations < 0)
  {
    throw std::runtime_error(
        "TimeObservationData received invalid dimensions");
  }
  num_levels_ = num_levels;
  num_obs_    = num_observations;
  data_.resize(num_levels_ * num_obs_);
  time_levels_ = Vector<Index>{};
  time_values_ = Vector<Real>{};
}

bool TimeObservationData::empty() const
{
  return data_.empty();
}

Index TimeObservationData::numLevels() const
{
  return num_levels_;
}

Index TimeObservationData::numObservations() const
{
  return num_obs_;
}

Index TimeObservationData::size() const
{
  return numLevels();
}

bool TimeObservationData::hasLayout() const
{
  return !sampler_.empty();
}

bool TimeObservationData::hasTimeLevels() const
{
  return !time_levels_.empty();
}

bool TimeObservationData::hasTimeValues() const
{
  return !time_values_.empty();
}

const std::string& TimeObservationData::sampler() const
{
  return sampler_;
}

const std::vector<Point3>& TimeObservationData::points() const
{
  return points_;
}

const Vector<Index>& TimeObservationData::components() const
{
  return components_;
}

const Vector<Index>& TimeObservationData::timeLevels() const
{
  return time_levels_;
}

const Vector<Real>& TimeObservationData::timeValues() const
{
  return time_values_;
}

Index TimeObservationData::timeLevel(Index row) const
{
  checkLevel(row);
  if (!hasTimeLevels())
  {
    return row;
  }
  return time_levels_[row];
}

Real TimeObservationData::timeValue(Index row) const
{
  checkLevel(row);
  if (!hasTimeValues())
  {
    return static_cast<Real>(timeLevel(row));
  }
  return time_values_[row];
}

void TimeObservationData::setLayout(std::string         sampler,
                                    std::vector<Point3> points,
                                    Vector<Index>       components)
{
  sampler_    = std::move(sampler);
  points_     = std::move(points);
  components_ = std::move(components);
  checkLayout();
}

void TimeObservationData::setTimeLevels(Vector<Index> levels)
{
  time_levels_ = std::move(levels);
  time_values_ = Vector<Real>{};
  checkTimeLevels();
}

void TimeObservationData::setTimeValues(Vector<Real> values)
{
  time_values_ = std::move(values);
  time_levels_ = Vector<Index>{};
  checkTimeValues();
}

Vector<Real> TimeObservationData::operator[](Index level)
{
  checkLevel(level);
  return Vector<Real>::view(data_.data() + level * num_obs_, num_obs_);
}

Vector<Real> TimeObservationData::operator[](Index level) const
{
  checkLevel(level);
  return Vector<Real>::view(
      const_cast<Real*>(data_.data()) + level * num_obs_, num_obs_);
}

void TimeObservationData::setZero()
{
  data_.setZero();
}

void TimeObservationData::checkLevel(Index level) const
{
  if (level < 0 || level >= numLevels())
  {
    throw std::runtime_error("TimeObservationData level is out of range");
  }
}

void TimeObservationData::checkLayout() const
{
  if (sampler_.empty())
  {
    return;
  }
  if (points_.empty() || components_.empty())
  {
    throw std::runtime_error(
        "TimeObservationData point layout is incomplete");
  }
  const Index expected =
      static_cast<Index>(points_.size()) * components_.size();
  if (expected != numObservations())
  {
    throw std::runtime_error(
        "TimeObservationData layout does not match observation count");
  }
}

void TimeObservationData::checkTimeLevels() const
{
  if (time_levels_.empty())
  {
    return;
  }
  if (time_levels_.size() != numLevels())
  {
    throw std::runtime_error(
        "TimeObservationData time level count does not match data");
  }
  for (Index i = 0; i < time_levels_.size(); ++i)
  {
    if (time_levels_[i] < 0
        || (i > 0 && time_levels_[i] <= time_levels_[i - 1]))
    {
      throw std::runtime_error(
          "TimeObservationData time levels must be strictly increasing");
    }
  }
}

void TimeObservationData::checkTimeValues() const
{
  if (time_values_.empty())
  {
    return;
  }
  if (time_values_.size() != numLevels())
  {
    throw std::runtime_error(
        "TimeObservationData time value count does not match data");
  }
  for (Index i = 0; i < time_values_.size(); ++i)
  {
    if (!std::isfinite(time_values_[i]) || time_values_[i] < 0.0
        || (i > 0 && time_values_[i] <= time_values_[i - 1]))
    {
      throw std::runtime_error(
          "TimeObservationData time values must be finite and increasing");
    }
  }
}

TimeObservationData sampleTimeObs(const TimeObservationOperator& obs,
                                  const solve::TimeTrajectory&   tr,
                                  const Vector<Real>&            prm)
{
  if (tr.numSteps() != obs.numSteps() || tr.numStates() != obs.numStates()
      || prm.size() != obs.numParams())
  {
    throw std::runtime_error("sampleTimeObs received inconsistent inputs");
  }

  TimeObservationData data(obs.numSteps() + 1, obs.numObservations());
  for (Index level = 0; level < data.numLevels(); ++level)
  {
    Vector<Real> values = data[level];
    obs.observe(level, tr[level], prm, values);
  }
  return data;
}

void writeTimeObsData(const std::string& path, const TimeObservationData& data)
{
  if (!data.hasLayout())
  {
    throw std::runtime_error(
        "Cannot write time observation data without point layout");
  }

  std::ofstream out(path);
  if (!out)
  {
    throw std::runtime_error("Failed to open time observation data file: "
                             + path);
  }

  out << std::setprecision(std::numeric_limits<Real>::max_digits10);
  out << "femx_time_obs_data\n\n";
  out << "num_levels " << data.numLevels() << "\n\n";
  if (data.hasTimeValues())
  {
    out << "time_values\n";
    for (Index row = 0; row < data.numLevels(); ++row)
    {
      out << "  " << data.timeValue(row) << '\n';
    }
    out << '\n';
  }
  else if (data.hasTimeLevels())
  {
    out << "time_levels\n";
    for (Index row = 0; row < data.numLevels(); ++row)
    {
      out << "  " << data.timeLevel(row) << '\n';
    }
    out << '\n';
  }

  out << "num_points " << static_cast<Index>(data.points().size()) << '\n';
  out << "points\n";
  for (const Point3& point : data.points())
  {
    out << "  " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
  }

  out << "\nnum_components " << data.components().size() << '\n';
  out << "components\n";
  for (Index component : data.components())
  {
    out << "  " << component << '\n';
  }

  out << "\nvalues\n";
  const Index num_components = data.components().size();
  const Index num_points     = static_cast<Index>(data.points().size());
  for (Index level = 0; level < data.numLevels(); ++level)
  {
    const Vector<Real> values = data[level];
    out << "  level " << level << '\n';
    for (Index point = 0; point < num_points; ++point)
    {
      out << "    ";
      for (Index component = 0; component < num_components; ++component)
      {
        if (component > 0)
        {
          out << ' ';
        }
        out << values[point * num_components + component];
      }
      out << '\n';
    }
    if (level + 1 < data.numLevels())
    {
      out << '\n';
    }
  }
}

namespace
{

void requireKey(const std::string& got,
                const std::string& expected)
{
  if (got != expected)
  {
    throw std::runtime_error("Time observation data missing " + expected);
  }
}

} // namespace

TimeObservationData readTimeObsData(const std::string& path)
{
  std::ifstream in(path);
  if (!in)
  {
    throw std::runtime_error("Failed to open time observation data file: "
                             + path);
  }

  std::string key;
  in >> key;
  requireKey(key, "femx_time_obs_data");

  Index num_levels = 0;
  in >> key >> num_levels;
  requireKey(key, "num_levels");

  Vector<Index> time_levels;
  Vector<Real>  time_values;
  in >> key;
  if (key == "time_values")
  {
    time_values.resize(num_levels);
    for (Index i = 0; i < num_levels; ++i)
    {
      in >> time_values[i];
    }
    in >> key;
  }
  else if (key == "time_levels")
  {
    time_levels.resize(num_levels);
    for (Index i = 0; i < num_levels; ++i)
    {
      in >> time_levels[i];
    }
    in >> key;
  }

  Index num_points = 0;
  requireKey(key, "num_points");
  in >> num_points;

  in >> key;
  requireKey(key, "points");
  std::vector<Point3> points;
  points.reserve(static_cast<std::size_t>(num_points));
  for (Index i = 0; i < num_points; ++i)
  {
    Point3 point{};
    in >> point[0] >> point[1] >> point[2];
    points.push_back(point);
  }

  Index num_components = 0;
  in >> key >> num_components;
  requireKey(key, "num_components");

  in >> key;
  requireKey(key, "components");
  Vector<Index> components(num_components);
  for (Index i = 0; i < num_components; ++i)
  {
    in >> components[i];
  }

  in >> key;
  requireKey(key, "values");

  TimeObservationData data(num_levels, num_points * num_components);
  data.setLayout("point", std::move(points), std::move(components));
  if (!time_values.empty())
  {
    data.setTimeValues(std::move(time_values));
  }
  else if (!time_levels.empty())
  {
    data.setTimeLevels(std::move(time_levels));
  }

  for (Index level = 0; level < num_levels; ++level)
  {
    Index label = 0;
    in >> key >> label;
    requireKey(key, "level");
    if (label != level)
    {
      throw std::runtime_error(
          "Time observation data has unexpected level label");
    }

    Vector<Real> values = data[level];
    for (Index i = 0; i < values.size(); ++i)
    {
      if (!(in >> values[i]))
      {
        throw std::runtime_error(
            "Time observation data ended before all values were read");
      }
    }
  }
  return data;
}

} // namespace problem
} // namespace femx
