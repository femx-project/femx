#include "TrajectoryObservation.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>

#include "../ns-var/Helper.hpp"
#include "ObservationVti.hpp"
#include <femx/common/LinearInterpolation.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/io/TimeSeriesDataIn.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>
#include <femx/problem/TimeObservationData.hpp>

namespace femx::make_obs
{
namespace
{

using namespace femx::problem;

constexpr unsigned long long kNoiseSeed = 0x5EED1234ULL;

struct NoiseReport
{
  bool  enabled    = false;
  Real  signal_rms = 0.0;
  Real  sigma      = 0.0;
  Index count      = 0;
};

void ensureParentDir(const std::string& path)
{
  const std::filesystem::path file(path);
  const std::filesystem::path dir = file.parent_path();
  if (!dir.empty())
  {
    std::filesystem::create_directories(dir);
  }
}

bool hasTimeSample(const TimeSampleParams& time)
{
  return time.start_time || time.end_time || time.start_level
         || time.end_level || time.num_points;
}

Vector<Real> trajectoryTimes(const TimeSeriesDataIn& trajectory)
{
  Vector<Real> times(trajectory.numSteps());
  for (Index step = 0; step < trajectory.numSteps(); ++step)
  {
    times[step] = trajectory.time(step);
    if (step > 0 && times[step] <= times[step - 1])
    {
      throw std::runtime_error(
          "trajectory times must be strictly increasing");
    }
  }
  return times;
}

Real startTime(const TimeSampleParams& time,
               const Vector<Real>&     trajectory_times)
{
  if (time.start_time)
  {
    return *time.start_time;
  }
  if (time.start_level)
  {
    if (*time.start_level >= trajectory_times.size())
    {
      throw std::runtime_error(
          "make_obs.time.start_level is outside the trajectory");
    }
    return trajectory_times[*time.start_level];
  }
  return trajectory_times.front();
}

Real endTime(const TimeSampleParams& time,
             const Vector<Real>&     trajectory_times)
{
  if (time.end_time)
  {
    return *time.end_time;
  }
  if (time.end_level)
  {
    if (*time.end_level >= trajectory_times.size())
    {
      throw std::runtime_error(
          "make_obs.time.end_level is outside the trajectory");
    }
    return trajectory_times[*time.end_level];
  }
  return trajectory_times.back();
}

void checkTimeRange(Real                start,
                    Real                end,
                    const Vector<Real>& trajectory_times)
{
  const Real max_time = trajectory_times.back();
  const Real tol =
      std::max<Real>(1.0e-10, 1.0e-8 * std::max<Real>(1.0, max_time));
  if (start < trajectory_times.front() - tol || end > max_time + tol)
  {
    throw std::runtime_error(
        "make_obs.time range is outside the trajectory");
  }
  if (start > end)
  {
    throw std::runtime_error("make_obs.time start must not exceed end");
  }
}

Vector<Real> selectedTimes(const TimeSampleParams& time,
                           const Vector<Real>&     trajectory_times)
{
  if (!hasTimeSample(time))
  {
    return trajectory_times;
  }

  const Real begin  = startTime(time, trajectory_times);
  const Real finish = endTime(time, trajectory_times);
  checkTimeRange(begin, finish, trajectory_times);

  if (!time.num_points)
  {
    Vector<Real> times;
    for (Real value : trajectory_times)
    {
      if (value >= begin - 1.0e-10 && value <= finish + 1.0e-10)
      {
        times.push_back(value);
      }
    }
    if (times.empty())
    {
      throw std::runtime_error("make_obs.time selected no trajectory levels");
    }
    return times;
  }

  const Index  count = *time.num_points;
  Vector<Real> times(count);
  if (count == 1)
  {
    times[0] = begin;
    return times;
  }

  const Index intervals = count - 1;
  for (Index i = 0; i < count; ++i)
  {
    times[i] = begin
               + (finish - begin) * static_cast<Real>(i)
                     / static_cast<Real>(intervals);
  }
  return times;
}

Vector<Real> relativeTimes(const Vector<Real>& absolute_times)
{
  Vector<Real> times(absolute_times.size());
  const Real   start = absolute_times.empty() ? 0.0 : absolute_times.front();
  for (Index i = 0; i < absolute_times.size(); ++i)
  {
    times[i] = absolute_times[i] - start;
    if (i == 0 && std::abs(times[i]) < 1.0e-14)
    {
      times[i] = 0.0;
    }
  }
  return times;
}

LinearInterpolation bracketTime(const Vector<Real>& trajectory_times,
                                Real                time)
{
  const Real tol =
      std::max<Real>(1.0e-10, 1.0e-8 * std::max<Real>(1.0, trajectory_times.back()));
  if (time < trajectory_times.front() - tol
      || time > trajectory_times.back() + tol)
  {
    throw std::runtime_error("observation time is outside the trajectory");
  }

  if (time <= trajectory_times.front() + tol)
  {
    return {0, 0, 0.0};
  }
  const Index last = trajectory_times.size() - 1;
  if (time >= trajectory_times.back() - tol)
  {
    return {last, last, 0.0};
  }

  const Real clamped =
      std::min<Real>(std::max<Real>(time, trajectory_times.front()),
                     trajectory_times.back());
  return linearInterpolation(trajectory_times, clamped);
}

std::array<Vector<Real>, 3> interpolateVelocity(
    const TimeSeriesDataIn& trajectory,
    const std::string&      field_name,
    const Vector<Real>&     trajectory_times,
    Real                    time)
{
  const LinearInterpolation interp = bracketTime(trajectory_times, time);
  const auto&               lower =
      trajectory.vectorField(interp.lower, field_name);

  std::array<Vector<Real>, 3> out{
      Vector<Real>(trajectory.mesh().numNodes()),
      Vector<Real>(trajectory.mesh().numNodes()),
      Vector<Real>(trajectory.mesh().numNodes())};

  if (!interp.hasUpper())
  {
    for (Index d = 0; d < 3; ++d)
    {
      out[static_cast<std::size_t>(d)] =
          lower[static_cast<std::size_t>(d)];
    }
    return out;
  }

  const auto& upper =
      trajectory.vectorField(interp.upper, field_name);
  for (Index d = 0; d < 3; ++d)
  {
    for (Index node = 0; node < trajectory.mesh().numNodes(); ++node)
    {
      out[static_cast<std::size_t>(d)][node] =
          interp.lowerWeight() * lower[static_cast<std::size_t>(d)][node]
          + interp.upperWeight() * upper[static_cast<std::size_t>(d)][node];
    }
  }
  return out;
}

MixedFESpace makeVelocitySpace(Mesh& mesh, FiniteElement& elem)
{
  FESpace      velocity(&mesh, &elem, mesh.dim());
  MixedFESpace space;
  space.addField(velocity);
  space.setup();
  return space;
}

Vector<Real> velocityState(
    const MixedFESpace&                space,
    const std::array<Vector<Real>, 3>& velocity)
{
  Vector<Real> state(space.numDofs());
  state.setZero();

  const auto field = space.field(0);
  for (Index node = 0; node < space.mesh().numNodes(); ++node)
  {
    for (Index component = 0; component < field.numComponents();
         ++component)
    {
      state[field.globalDof(node, component)] =
          velocity[static_cast<std::size_t>(component)][node];
    }
  }
  return state;
}

Real snrAmplitudeRatio(const NoiseParams& noise)
{
  if (noise.snr)
  {
    return *noise.snr;
  }
  return std::pow(10.0, *noise.snr_db / 20.0);
}

NoiseReport addGaussianNoise(TimeObservationData& data,
                             const NoiseParams&   noise)
{
  NoiseReport report;
  if (!noise.enabled)
  {
    return report;
  }

  Real  sum_sq = 0.0;
  Index count  = 0;
  for (Index level = 0; level < data.numLevels(); ++level)
  {
    const Vector<Real> values = data[level];
    for (Index i = 0; i < values.size(); ++i)
    {
      sum_sq += values[i] * values[i];
      ++count;
    }
  }
  if (count == 0)
  {
    throw std::runtime_error("Cannot add noise to empty observation data");
  }

  report.enabled    = true;
  report.signal_rms = std::sqrt(sum_sq / static_cast<Real>(count));
  report.count      = count;
  if (report.signal_rms <= 0.0)
  {
    throw std::runtime_error("Cannot set SNR for zero observation signal");
  }
  report.sigma = report.signal_rms / snrAmplitudeRatio(noise);

  std::mt19937_64                rng(kNoiseSeed);
  std::normal_distribution<Real> dist(0.0, report.sigma);
  for (Index level = 0; level < data.numLevels(); ++level)
  {
    Vector<Real> values = data[level];
    for (Index i = 0; i < values.size(); ++i)
    {
      values[i] += dist(rng);
    }
  }
  return report;
}

void writeSummary(const Params&              prm,
                  const std::string&         observation_name,
                  const TimeObservationData& data,
                  const NoiseReport&         noise,
                  const std::string&         vti_output)
{
  std::cout << "\nFinal summary\n";
  std::cout << "  trajectory: " << prm.input.trajectory << '\n';
  if (!observation_name.empty())
  {
    std::cout << "  observation: " << observation_name << '\n';
  }
  std::cout << "  levels: " << data.numLevels()
            << ", observations/level: " << data.numObservations() << '\n';
  std::cout << "  output: " << prm.output.file << '\n';
  if (!vti_output.empty())
  {
    std::cout << "  vti output: " << vti_output << '\n';
  }
  if (prm.output.write_reference && !prm.output.reference_basename.empty())
  {
    std::cout << "  reference output: "
              << prm.output.reference_basename << ".xdmf\n";
  }
  if (noise.enabled)
  {
    std::cout << "  noise: gaussian, signal_rms = " << noise.signal_rms
              << ", sigma = " << noise.sigma
              << ", sampled values = " << noise.count
              << ", seed = " << kNoiseSeed << '\n';
  }
  else
  {
    std::cout << "  noise: disabled\n";
  }
}

void writeTrajectoryObservationOutput(
    const Params&           prm,
    const std::string&      observation_name,
    const TimeSeriesDataIn& trajectory,
    Mesh&                   mesh,
    const MixedFESpace&     space,
    const Vector<Real>&     trajectory_times)
{
  const Vector<Real> sample_times =
      selectedTimes(prm.time, trajectory_times);

  auto obs = navier_var_new::makeObs(space,
                                     prm.obs,
                                     std::max<Index>(sample_times.size() - 1, 0),
                                     space.numDofs(),
                                     0);

  TimeObservationData sampled(sample_times.size(), obs->numObservations());
  navier_var_new::setObsLayout(sampled, space, prm.obs);
  sampled.setTimeValues(relativeTimes(sample_times));

  TimeSeriesDataOut reference_out;
  if (prm.output.write_reference && !prm.output.reference_basename.empty())
  {
    reference_out.attachMesh(mesh);
  }

  std::cout << "  sample observations\n";
  Vector<Real> empty_prm;
  for (Index row = 0; row < sample_times.size(); ++row)
  {
    const auto         velocity = interpolateVelocity(trajectory,
                                              prm.input.velocity_field,
                                              trajectory_times,
                                              sample_times[row]);
    const Vector<Real> state    = velocityState(space, velocity);

    Vector<Real> values;
    obs->observe(row, state, empty_prm, values);
    Vector<Real> dst = sampled[row];
    for (Index i = 0; i < values.size(); ++i)
    {
      dst[i] = values[i];
    }

    if (prm.output.write_reference
        && !prm.output.reference_basename.empty())
    {
      reference_out.beginStep(sample_times[row]);
      reference_out.addNodalVectorField(
          "velocity", velocity[0], velocity[1], velocity[2]);
    }
  }

  TimeObservationData noisy = sampled;
  const NoiseReport   noise = addGaussianNoise(noisy, prm.noise);

  ensureParentDir(prm.output.file);
  writeTimeObsData(prm.output.file, noisy);
  const std::string vti_output = writeObservationVtiOutputs(prm, noisy);

  if (prm.output.write_reference && !prm.output.reference_basename.empty())
  {
    ensureParentDir(prm.output.reference_basename + ".h5");
    reference_out.write(prm.output.reference_basename);
  }

  writeSummary(prm, observation_name, noisy, noise, vti_output);
}

Params observationParams(const Params&          base,
                         const ObservationCase& item)
{
  Params out = base;
  out.obs    = item.obs;
  out.output = item.output;
  out.noise  = item.noise;
  out.time   = item.time;
  out.observations.clear();
  return out;
}

} // namespace

void writeTrajectoryObservationOutputs(const Params& prm)
{
  if (prm.input.trajectory.empty())
  {
    throw std::runtime_error(
        "make_obs.input.trajectory is required; run a forward solver first "
        "and pass the velocity XDMF/HDF5 time series to make-obs");
  }
  std::cout << "  read trajectory\n";
  const TimeSeriesDataIn trajectory =
      TimeSeriesDataIn::read(prm.input.trajectory);
  if (trajectory.numSteps() <= 0)
  {
    throw std::runtime_error("trajectory has no time steps");
  }

  Mesh         mesh  = trajectory.mesh();
  auto         elem  = navier_var_new::makeElement(mesh);
  MixedFESpace space = makeVelocitySpace(mesh, *elem);

  const Vector<Real> trajectory_times = trajectoryTimes(trajectory);

  if (prm.observations.empty())
  {
    writeTrajectoryObservationOutput(
        prm, {}, trajectory, mesh, space, trajectory_times);
    return;
  }

  for (const ObservationCase& item : prm.observations)
  {
    Params local = observationParams(prm, item);
    writeTrajectoryObservationOutput(
        local, item.name, trajectory, mesh, space, trajectory_times);
  }
}

} // namespace femx::make_obs
