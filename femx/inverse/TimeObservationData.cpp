#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/inverse/TimeObservationData.hpp>
#include <femx/inverse/TimeObservationOperator.hpp>
#include <femx/state/TimeTrajectory.hpp>
using namespace femx::state;

namespace femx
{
namespace inverse
{

TimeObservationData::TimeObservationData(Index num_levels,
                                         Index num_obs)
{
  resize(num_levels, num_obs);
}

void TimeObservationData::resize(Index num_levels, Index num_obs)
{
  if (num_levels < 0 || num_obs < 0)
  {
    throw std::runtime_error(
        "TimeObservationData received invalid dimensions");
  }
  num_levels_ = num_levels;
  num_obs_    = num_obs;
  data_.resize(num_levels_ * num_obs_);
  sampler_.clear();
  pts_         = Array<Point3>{};
  comps_       = Array<Index>{};
  time_levels_ = Array<Index>{};
  time_vals_   = HostVector{};
}

bool TimeObservationData::empty() const
{
  return data_.empty();
}

Index TimeObservationData::numTimeLevels() const
{
  return num_levels_;
}

Index TimeObservationData::numObservations() const
{
  return num_obs_;
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
  return !time_vals_.empty();
}

const std::string& TimeObservationData::sampler() const
{
  return sampler_;
}

const Array<Point3>& TimeObservationData::pts() const
{
  return pts_;
}

const Array<Index>& TimeObservationData::comps() const
{
  return comps_;
}

const Array<Index>& TimeObservationData::timeLevels() const
{
  return time_levels_;
}

const HostVector& TimeObservationData::timeValues() const
{
  return time_vals_;
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
  return time_vals_[row];
}

void TimeObservationData::setLayout(std::string   sampler,
                                    Array<Point3> pts,
                                    Array<Index>  comps)
{
  sampler_ = std::move(sampler);
  pts_     = std::move(pts);
  comps_   = std::move(comps);
  checkLayout();
}

void TimeObservationData::setTimeLevels(Array<Index> levels)
{
  time_levels_ = std::move(levels);
  time_vals_   = HostVector{};
  checkTimeLevels();
}

void TimeObservationData::setTimeValues(HostVector vals)
{
  time_vals_   = std::move(vals);
  time_levels_ = Array<Index>{};
  checkTimeValues();
}

HostVectorView TimeObservationData::operator[](Index level)
{
  checkLevel(level);
  return HostVectorView(data_.data() + level * num_obs_, num_obs_);
}

HostConstVectorView TimeObservationData::operator[](Index level) const
{
  checkLevel(level);
  return HostConstVectorView(data_.data() + level * num_obs_, num_obs_);
}

void TimeObservationData::setZero()
{
  data_.setZero();
}

void TimeObservationData::checkLevel(Index level) const
{
  if (level < 0 || level >= numTimeLevels())
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
  if (pts_.empty() || comps_.empty())
  {
    throw std::runtime_error(
        "TimeObservationData point layout is incomplete");
  }
  const Index exp = pts_.size() * comps_.size();
  if (exp != numObservations())
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
  if (time_levels_.size() != numTimeLevels())
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
  if (time_vals_.empty())
  {
    return;
  }
  if (time_vals_.size() != numTimeLevels())
  {
    throw std::runtime_error(
        "TimeObservationData time value count does not match data");
  }
  for (Index i = 0; i < time_vals_.size(); ++i)
  {
    if (!std::isfinite(time_vals_[i]) || time_vals_[i] < 0.0
        || (i > 0 && time_vals_[i] <= time_vals_[i - 1]))
    {
      throw std::runtime_error(
          "TimeObservationData time values must be finite and increasing");
    }
  }
}

TimeObservationData sampleTimeObs(const TimeObservationOperator& obs,
                                  const TimeTrajectory&          tr,
                                  const HostVector&              prm)
{
  if (tr.numSteps() != obs.numSteps() || tr.numStates() != obs.numStates()
      || prm.size() != obs.numParams())
  {
    throw std::runtime_error("sampleTimeObs received inconsistent inputs");
  }

  TimeObservationData data(obs.numSteps() + 1, obs.numObservations());
  for (Index level = 0; level < data.numTimeLevels(); ++level)
  {
    HostVector vals(obs.numObservations());
    obs.observe(level, tr[level], prm, vals);
    data[level] = vals;
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
  out << "num_levels " << data.numTimeLevels() << "\n\n";
  if (data.hasTimeValues())
  {
    out << "time_values\n";
    for (Index row = 0; row < data.numTimeLevels(); ++row)
    {
      out << "  " << data.timeValue(row) << '\n';
    }
    out << '\n';
  }
  else if (data.hasTimeLevels())
  {
    out << "time_levels\n";
    for (Index row = 0; row < data.numTimeLevels(); ++row)
    {
      out << "  " << data.timeLevel(row) << '\n';
    }
    out << '\n';
  }

  out << "num_points " << data.pts().size() << '\n';
  out << "points\n";
  for (const Point3& point : data.pts())
  {
    out << "  " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
  }

  out << "\nnum_comp " << data.comps().size() << '\n';
  out << "components\n";
  for (Index comp : data.comps())
  {
    out << "  " << comp << '\n';
  }

  out << "\nvalues\n";
  const Index num_comp   = data.comps().size();
  const Index num_points = data.pts().size();
  for (Index level = 0; level < data.numTimeLevels(); ++level)
  {
    const HostVector vals = data[level];
    out << "  level " << level << '\n';
    for (Index point = 0; point < num_points; ++point)
    {
      out << "    ";
      for (Index comp = 0; comp < num_comp; ++comp)
      {
        if (comp > 0)
        {
          out << ' ';
        }
        out << vals[point * num_comp + comp];
      }
      out << '\n';
    }
    if (level + 1 < data.numTimeLevels())
    {
      out << '\n';
    }
  }
}

namespace
{

void requireKey(const std::string& got,
                const std::string& exp)
{
  if (got != exp)
  {
    throw std::runtime_error("Time observation data missing " + exp);
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

  Array<Index> time_levels;
  HostVector   time_vals;
  in >> key;
  if (key == "time_values")
  {
    time_vals.resize(num_levels);
    for (Index i = 0; i < num_levels; ++i)
    {
      in >> time_vals[i];
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
  Array<Point3> pts;
  pts.reserve(num_points);
  for (Index i = 0; i < num_points; ++i)
  {
    Point3 point{};
    in >> point[0] >> point[1] >> point[2];
    pts.push_back(point);
  }

  Index num_comp = 0;
  in >> key >> num_comp;
  requireKey(key, "num_comp");

  in >> key;
  requireKey(key, "components");
  Array<Index> comps(num_comp);
  for (Index i = 0; i < num_comp; ++i)
  {
    in >> comps[i];
  }

  in >> key;
  requireKey(key, "values");

  TimeObservationData data(num_levels, num_points * num_comp);
  data.setLayout("point", std::move(pts), std::move(comps));
  if (!time_vals.empty())
  {
    data.setTimeValues(std::move(time_vals));
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

    HostVectorView vals = data[level];
    for (Index i = 0; i < vals.size(); ++i)
    {
      if (!(in >> vals[i]))
      {
        throw std::runtime_error(
            "Time observation data ended before all values were read");
      }
    }
  }
  return data;
}

} // namespace inverse
} // namespace femx
