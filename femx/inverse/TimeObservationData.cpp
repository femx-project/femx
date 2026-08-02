#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/common/Checks.hpp>
#include <femx/inverse/TimeObservationData.hpp>
#include <femx/inverse/TimeObservationOperator.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/host/HostContext.hpp>
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
  require(num_levels >= 0 && num_obs >= 0,
          "TimeObservationData received invalid dimensions");
  num_levels_ = num_levels;
  num_obs_    = num_obs;
  data_.assign(num_levels_ * num_obs_, 0);
  sampler_.clear();
  pts_         = HostVector<Point3>{};
  comps_       = HostVector<Index>{};
  time_levels_ = HostVector<Index>{};
  time_vals_   = HostVector<Real>{};
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

const HostVector<Point3>& TimeObservationData::pts() const
{
  return pts_;
}

const HostVector<Index>& TimeObservationData::comps() const
{
  return comps_;
}

const HostVector<Index>& TimeObservationData::timeLevels() const
{
  return time_levels_;
}

const HostVector<Real>& TimeObservationData::timeValues() const
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

void TimeObservationData::setLayout(std::string        sampler,
                                    HostVector<Point3> pts,
                                    HostVector<Index>  comps)
{
  sampler_ = std::move(sampler);
  pts_     = std::move(pts);
  comps_   = std::move(comps);
  checkLayout();
}

void TimeObservationData::setTimeLevels(HostVector<Index> levels)
{
  time_levels_ = std::move(levels);
  time_vals_   = HostVector<Real>{};
  checkTimeLevels();
}

void TimeObservationData::setTimeValues(HostVector<Real> vals)
{
  time_vals_   = std::move(vals);
  time_levels_ = HostVector<Index>{};
  checkTimeValues();
}

HostVectorView<Real> TimeObservationData::operator[](Index level)
{
  checkLevel(level);
  return HostVectorView<Real>(data_.data() + level * num_obs_, num_obs_);
}

HostVectorView<const Real> TimeObservationData::operator[](Index level) const
{
  checkLevel(level);
  return HostVectorView<const Real>(data_.data() + level * num_obs_, num_obs_);
}

void TimeObservationData::setZero()
{
  linalg::HostContext ctx;
  auto&               vec_handler = ctx.vectorHandler();
  vec_handler.zero(data_.view());
}

void TimeObservationData::checkLevel(Index level) const
{
  require(level >= 0 && level < numTimeLevels(),
          "TimeObservationData level is out of range");
}

void TimeObservationData::checkLayout() const
{
  if (sampler_.empty())
  {
    return;
  }
  require(!pts_.empty() && !comps_.empty(),
          "TimeObservationData point layout is incomplete");
  const Index exp = pts_.size() * comps_.size();
  require(exp == numObservations(),
          "TimeObservationData layout does not match observation count");
}

void TimeObservationData::checkTimeLevels() const
{
  if (time_levels_.empty())
  {
    return;
  }
  require(time_levels_.size() == numTimeLevels(),
          "TimeObservationData time level count does not match data");
  for (Index i = 0; i < time_levels_.size(); ++i)
  {
    require(time_levels_[i] >= 0
                && (i == 0 || time_levels_[i] > time_levels_[i - 1]),
            "TimeObservationData time levels must be strictly increasing");
  }
}

void TimeObservationData::checkTimeValues() const
{
  if (time_vals_.empty())
  {
    return;
  }
  require(time_vals_.size() == numTimeLevels(),
          "TimeObservationData time value count does not match data");
  for (Index i = 0; i < time_vals_.size(); ++i)
  {
    require(std::isfinite(time_vals_[i]) && time_vals_[i] >= 0.0
                && (i == 0 || time_vals_[i] > time_vals_[i - 1]),
            "TimeObservationData time values must be finite and increasing");
  }
}

TimeObservationData sampleTimeObs(const TimeObservationOperator& obs,
                                  const TimeTrajectory&          traj,
                                  const HostVector<Real>&        prm)
{
  require(traj.numSteps() == obs.numSteps()
              && traj.numStates() == obs.numStates()
              && prm.size() == obs.numParams(),
          "sampleTimeObs received inconsistent inputs");

  TimeObservationData data(obs.numSteps() + 1, obs.numObservations());
  for (Index level = 0; level < data.numTimeLevels(); ++level)
  {
    HostVector<Real> vals(obs.numObservations());
    obs.observe(level, traj[level], prm, vals);
    data[level] = vals;
  }
  return data;
}

void writeTimeObsData(const std::string& path, const TimeObservationData& data)
{
  require(data.hasLayout(),
          "Cannot write time observation data without point layout");

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
    for (Index i = 0; i < data.numTimeLevels(); ++i)
    {
      out << "  " << data.timeValue(i) << '\n';
    }
    out << '\n';
  }
  else if (data.hasTimeLevels())
  {
    out << "time_levels\n";
    for (Index i = 0; i < data.numTimeLevels(); ++i)
    {
      out << "  " << data.timeLevel(i) << '\n';
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
  for (Index ic : data.comps())
  {
    out << "  " << ic << '\n';
  }

  out << "\nvalues\n";
  const Index num_comp   = data.comps().size();
  const Index num_points = data.pts().size();
  for (Index level = 0; level < data.numTimeLevels(); ++level)
  {
    const HostVector<Real> vals = data[level];
    out << "  level " << level << '\n';
    for (Index point = 0; point < num_points; ++point)
    {
      out << "    ";
      for (Index ic = 0; ic < num_comp; ++ic)
      {
        if (ic > 0)
        {
          out << ' ';
        }
        out << vals[point * num_comp + ic];
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

  HostVector<Index> time_levels;
  HostVector<Real>  time_vals;
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
  HostVector<Point3> pts;
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
  HostVector<Index> comps(num_comp);
  for (Index ic = 0; ic < num_comp; ++ic)
  {
    in >> comps[ic];
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

    HostVectorView<Real> vals = data[level];
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
