#include <cmath>
#include <exception>
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

void TimeObservationData::resize(Index num_levels,
                                 Index num_observations)
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
  if (points_.empty())
  {
    throw std::runtime_error(
        "TimeObservationData layout requires at least one point");
  }
  if (components_.empty())
  {
    throw std::runtime_error(
        "TimeObservationData layout requires at least one component");
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
    if (time_levels_[i] < 0)
    {
      throw std::runtime_error(
          "TimeObservationData time levels must be non-negative");
    }
    if (i > 0 && time_levels_[i] <= time_levels_[i - 1])
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
    if (!std::isfinite(time_values_[i]) || time_values_[i] < 0.0)
    {
      throw std::runtime_error(
          "TimeObservationData time values must be finite and non-negative");
    }
    if (i > 0 && time_values_[i] <= time_values_[i - 1])
    {
      throw std::runtime_error(
          "TimeObservationData time values must be strictly increasing");
    }
  }
}

void writeTimeObsData(const std::string&         path,
                      const TimeObservationData& data)
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

  const Index num_points =
      static_cast<Index>(data.points().size());
  const Index num_components = data.components().size();

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

  out << "num_points " << num_points << '\n';
  out << "points\n";
  for (const Point3& point : data.points())
  {
    out << "  " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
  }

  out << "\nnum_components " << num_components << '\n';
  out << "components\n";
  for (Index component : data.components())
  {
    out << "  " << component << '\n';
  }

  out << "\nvalues\n";
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

bool parseIndexToken(const std::string& token,
                     Index&             value)
{
  try
  {
    std::size_t     parsed = 0;
    const long long raw    = std::stoll(token, &parsed);
    if (parsed != token.size()
        || raw < static_cast<long long>(std::numeric_limits<Index>::min())
        || raw > static_cast<long long>(std::numeric_limits<Index>::max()))
    {
      return false;
    }
    value = static_cast<Index>(raw);
    return true;
  }
  catch (const std::exception&)
  {
    return false;
  }
}

TimeObservationData readVersionedTimeObsData(std::istream&      in,
                                             const std::string& path,
                                             Index              version)
{
  if (version < 1 || version > 4)
  {
    throw std::runtime_error("Unsupported time observation data file: "
                             + path);
  }

  std::string key;
  Index       num_levels = 0;
  Index       num_obs    = 0;

  if (version == 1)
  {
    in >> key >> num_levels;
    if (key != "num_levels")
    {
      throw std::runtime_error("Time observation data missing num_levels");
    }
    in >> key >> num_obs;
    if (key != "num_observations")
    {
      throw std::runtime_error(
          "Time observation data missing num_observations");
    }
    in >> key;
    if (key != "values")
    {
      throw std::runtime_error("Time observation data missing values block");
    }

    TimeObservationData data(num_levels, num_obs);
    for (Index level = 0; level < num_levels; ++level)
    {
      Vector<Real> values = data[level];
      for (Index i = 0; i < num_obs; ++i)
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

  std::string sampler;
  in >> key >> sampler;
  if (key != "sampler")
  {
    throw std::runtime_error("Time observation data missing sampler");
  }

  in >> key >> num_levels;
  if (key != "num_levels")
  {
    throw std::runtime_error("Time observation data missing num_levels");
  }

  Vector<Index> time_levels;
  Vector<Real>  time_values;
  if (version >= 3)
  {
    in >> key;
    if (version == 3 && key != "time_levels")
    {
      throw std::runtime_error(
          "Time observation data missing time_levels");
    }
    if (version == 4 && key != "time_values")
    {
      throw std::runtime_error(
          "Time observation data missing time_values");
    }
    if (version == 3)
    {
      time_levels.resize(num_levels);
      for (Index row = 0; row < num_levels; ++row)
      {
        if (!(in >> time_levels[row]))
        {
          throw std::runtime_error(
              "Time observation data ended before all time levels were read");
        }
      }
    }
    else
    {
      time_values.resize(num_levels);
      for (Index row = 0; row < num_levels; ++row)
      {
        if (!(in >> time_values[row]))
        {
          throw std::runtime_error(
              "Time observation data ended before all time values were read");
        }
      }
    }
  }

  Index num_points = 0;
  in >> key >> num_points;
  if (key != "num_points")
  {
    throw std::runtime_error("Time observation data missing num_points");
  }
  Index num_components = 0;
  in >> key >> num_components;
  if (key != "num_components")
  {
    throw std::runtime_error("Time observation data missing num_components");
  }

  in >> key;
  if (key != "components")
  {
    throw std::runtime_error("Time observation data missing components");
  }
  Vector<Index> components(num_components);
  for (Index i = 0; i < num_components; ++i)
  {
    if (!(in >> components[i]))
    {
      throw std::runtime_error(
          "Time observation data ended before all components were read");
    }
  }

  in >> key;
  if (key != "points")
  {
    throw std::runtime_error("Time observation data missing points block");
  }
  std::vector<Point3> points;
  points.reserve(static_cast<std::size_t>(num_points));
  for (Index i = 0; i < num_points; ++i)
  {
    Point3 point{};
    if (!(in >> point[0] >> point[1] >> point[2]))
    {
      throw std::runtime_error(
          "Time observation data ended before all points were read");
    }
    points.push_back(point);
  }

  in >> key >> num_obs;
  if (key != "num_observations")
  {
    throw std::runtime_error(
        "Time observation data missing num_observations");
  }
  in >> key;
  if (key != "values")
  {
    throw std::runtime_error("Time observation data missing values block");
  }

  TimeObservationData data(num_levels, num_obs);
  data.setLayout(std::move(sampler), std::move(points), std::move(components));
  if (version == 3)
  {
    data.setTimeLevels(std::move(time_levels));
  }
  else if (version == 4)
  {
    data.setTimeValues(std::move(time_values));
  }
  for (Index level = 0; level < num_levels; ++level)
  {
    Vector<Real> values = data[level];
    for (Index i = 0; i < num_obs; ++i)
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

TimeObservationData readCurrentTimeObsData(std::istream&      in,
                                           const std::string& path,
                                           std::string        key)
{
  Index num_levels = 0;
  if (key != "num_levels" || !(in >> num_levels))
  {
    throw std::runtime_error("Time observation data missing num_levels");
  }

  Vector<Index> time_levels;
  Vector<Real>  time_values;
  in >> key;
  if (key == "time_values")
  {
    time_values.resize(num_levels);
    for (Index row = 0; row < num_levels; ++row)
    {
      if (!(in >> time_values[row]))
      {
        throw std::runtime_error(
            "Time observation data ended before all time values were read");
      }
    }
    in >> key;
  }
  else if (key == "time_levels")
  {
    time_levels.resize(num_levels);
    for (Index row = 0; row < num_levels; ++row)
    {
      if (!(in >> time_levels[row]))
      {
        throw std::runtime_error(
            "Time observation data ended before all time levels were read");
      }
    }
    in >> key;
  }

  Index num_points = 0;
  if (key != "num_points" || !(in >> num_points))
  {
    throw std::runtime_error("Time observation data missing num_points");
  }

  in >> key;
  if (key != "points")
  {
    throw std::runtime_error("Time observation data missing points block");
  }
  std::vector<Point3> points;
  points.reserve(static_cast<std::size_t>(num_points));
  for (Index i = 0; i < num_points; ++i)
  {
    Point3 point{};
    if (!(in >> point[0] >> point[1] >> point[2]))
    {
      throw std::runtime_error(
          "Time observation data ended before all points were read");
    }
    points.push_back(point);
  }

  Index num_components = 0;
  in >> key >> num_components;
  if (key != "num_components")
  {
    throw std::runtime_error("Time observation data missing num_components");
  }

  in >> key;
  if (key != "components")
  {
    throw std::runtime_error("Time observation data missing components");
  }
  Vector<Index> components(num_components);
  for (Index i = 0; i < num_components; ++i)
  {
    if (!(in >> components[i]))
    {
      throw std::runtime_error(
          "Time observation data ended before all components were read");
    }
  }

  in >> key;
  if (key != "values")
  {
    throw std::runtime_error("Time observation data missing values block");
  }

  const Index         num_obs = num_points * num_components;
  TimeObservationData data(num_levels, num_obs);
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
    Index level_label = 0;
    in >> key >> level_label;
    if (key != "level" || level_label != level)
    {
      throw std::runtime_error(
          "Time observation data missing level block");
    }

    Vector<Real> values = data[level];
    for (Index point = 0; point < num_points; ++point)
    {
      for (Index component = 0; component < num_components; ++component)
      {
        if (!(in >> values[point * num_components + component]))
        {
          throw std::runtime_error(
              "Time observation data ended before all values were read");
        }
      }
    }
  }
  return data;
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

  std::string tag;
  in >> tag;
  if (tag != "femx_time_obs_data")
  {
    throw std::runtime_error("Unsupported time observation data file: "
                             + path);
  }

  std::string key;
  if (!(in >> key))
  {
    throw std::runtime_error("Time observation data ended after header");
  }

  Index version = 0;
  if (parseIndexToken(key, version))
  {
    return readVersionedTimeObsData(in, path, version);
  }

  return readCurrentTimeObsData(in, path, key);
}

} // namespace problem
} // namespace femx
