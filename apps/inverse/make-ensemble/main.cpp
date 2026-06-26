#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include <Common.hpp>
#include <Helper.hpp>
#include <femx/common/Math.hpp>
#include <femx/fem/VelocityProfile.hpp>
#include <femx/linalg/DenseMatrix.hpp>

using namespace std;
using namespace femx;
using namespace femx::navier_en_var;
namespace inv = femx::inverse;

#ifndef FEMX_MAKE_ENSEMBLE_APP_NAME
#define FEMX_MAKE_ENSEMBLE_APP_NAME "make-ensemble"
#endif

namespace
{

struct AppOptions
{
  string          config_file;
  string          output_file;
  string          mean_output_file;
  optional<Index> modes;
  optional<Index> initial_modes;
  Real            amplitude = 0.35;
  bool            help      = false;
};

AppOptions parseAppOptions(int argc, char** argv)
{
  AppOptions opts;
  for (int i = 1; i < argc; ++i)
  {
    const string key(argv[i]);
    if (key == "-h" || key == "--help")
    {
      opts.help = true;
      return opts;
    }
    if (key == "--config" || key == "-config")
    {
      opts.config_file = inv::requireValue(argc, argv, i, key);
      continue;
    }
    if (key == "--output")
    {
      opts.output_file = inv::requireValue(argc, argv, i, key);
      continue;
    }
    if (key == "--mean-output")
    {
      opts.mean_output_file = inv::requireValue(argc, argv, i, key);
      continue;
    }
    if (key == "--modes")
    {
      opts.modes = static_cast<Index>(
          stoi(inv::requireValue(argc, argv, i, key)));
      if (*opts.modes < 0)
      {
        throw runtime_error("--modes must be nonnegative");
      }
      continue;
    }
    if (key == "--initial-modes")
    {
      opts.initial_modes = static_cast<Index>(
          stoi(inv::requireValue(argc, argv, i, key)));
      if (*opts.initial_modes < 0)
      {
        throw runtime_error("--initial-modes must be nonnegative");
      }
      continue;
    }
    if (key == "--amplitude")
    {
      opts.amplitude = stod(inv::requireValue(argc, argv, i, key));
      if (!isfinite(opts.amplitude) || opts.amplitude < 0.0)
      {
        throw runtime_error("--amplitude must be nonnegative");
      }
      continue;
    }
    throw runtime_error("Unknown option: " + key);
  }
  return opts;
}

void printUsage(ostream& out)
{
  out << "Usage: " << FEMX_MAKE_ENSEMBLE_APP_NAME
      << " --config FILE [--output FILE] [--mean-output FILE]"
      << " [--modes N] [--initial-modes N] [--amplitude A]\n";
  out << "  Output defaults to inverse.ensemble.perturbations_file when set.\n";
  out << "  Mean output defaults to inverse.ensemble.mean_file when set.\n";
  out << "  --modes is the number of boundary pulse modes.\n";
  out << "  --initial-modes is the number of initial velocity modes.\n";
  out << "  --amplitude is the relative perturbation amplitude.\n";
}

Index defaultBoundaryModeCount(const InverseParameterLayout& lyt)
{
  return min<Index>(max<Index>(lyt.nctr, 1), 4);
}

Index defaultInitialModeCount(const AppNsEnVar& app)
{
  return app.lyt.hasInitialVelocity() ? Index{2} : Index{0};
}

void fillBoundaryPulsePerturbations(DenseMatrix&      out,
                                    const AppNsEnVar& app,
                                    const Params&     prm,
                                    Index             col_offset,
                                    Index             modes,
                                    Real              amplitude)
{
  const auto& target = controlTarget(prm);
  const auto  prof =
      fem::poiseuilleProfile(target.cen, target.nrm, target.rad);
  const auto  u_fe      = app.space.field(0);
  const Index nd        = u_fe.numComponents();
  const Index ncd       = app.ctr.numDofs();
  const Real  pi        = acos(-1.0);
  const Real  base_peak = fem::peakSpeed(
      target.qty, "poiseuille", target.bulk_speed, 1.0, 1.5);

  for (Index level = 0; level < app.lyt.nctr; ++level)
  {
    const Real time = app.ctr_times[level];
    for (Index m = 0; m < modes; ++m)
    {
      const Real temporal =
          sin(2.0 * pi * static_cast<Real>(m + 1) * time
              / target.per);
      const Real peak = amplitude * base_peak * temporal;
      for (Index i = 0; i < ncd; ++i)
      {
        const Index id   = app.ctr.stateDof(i);
        const Index in   = nd > 0 ? id / nd : 0;
        const Index comp = nd > 0 ? id - nd * in : 0;
        const Index row  = app.lyt.coff + level * ncd + i;
        out(row, col_offset + m) =
            comp >= 0 && comp < 3
                ? fem::velocityComponent(
                      prof, app.space.mesh().node(in), peak, comp)
                : 0.0;
      }
    }
  }
}

struct AxialBounds
{
  Real lower = 0.0;
  Real upper = 0.0;
};

AxialBounds axialBounds(const Mesh&   mesh,
                        const Point3& axis)
{
  if (mesh.numNodes() <= 0)
  {
    throw runtime_error("initial velocity mode generation requires mesh nodes");
  }

  AxialBounds bounds;
  bounds.lower = dot(mesh.node(0), axis);
  bounds.upper = bounds.lower;
  for (Index in = 1; in < mesh.numNodes(); ++in)
  {
    const Real x = dot(mesh.node(in), axis);
    bounds.lower = min(bounds.lower, x);
    bounds.upper = max(bounds.upper, x);
  }
  return bounds;
}

Real axialCoordinate(const Point3&      point,
                     const Point3&      axis,
                     const AxialBounds& bounds)
{
  const Real len = bounds.upper - bounds.lower;
  if (len <= 0.0)
  {
    return 0.0;
  }
  const Real s = (dot(point, axis) - bounds.lower) / len;
  return min(Real{1.0}, max(Real{0.0}, s));
}

void fillInitialVelocityPerturbations(DenseMatrix&      out,
                                      const AppNsEnVar& app,
                                      const Params&     prm,
                                      Index             col_offset,
                                      Index             modes,
                                      Real              amplitude)
{
  if (modes <= 0)
  {
    return;
  }
  if (!app.lyt.hasInitialVelocity())
  {
    throw runtime_error(
        "--initial-modes requires controls.init_vel.enabled=true");
  }

  const auto& target = controlTarget(prm);
  const auto  prof =
      fem::poiseuilleProfile(target.cen, target.nrm, target.rad);
  const auto  axis      = unit(target.nrm);
  const auto  bounds    = axialBounds(app.space.mesh(), axis);
  const auto  u_fe      = app.space.field(0);
  const Index nd        = u_fe.numComponents();
  const Real  pi        = acos(-1.0);
  const Real  base_peak = fem::peakSpeed(
      target.qty, "poiseuille", target.bulk_speed, 1.0, 1.5);

  for (Index m = 0; m < modes; ++m)
  {
    const Index col = col_offset + m;
    for (Index i = 0; i < app.init_vdofs.size(); ++i)
    {
      const Index id   = app.init_vdofs[i];
      const Index in   = nd > 0 ? id / nd : 0;
      const Index comp = nd > 0 ? id - nd * in : 0;
      if (in < 0 || in >= app.space.mesh().numNodes())
      {
        throw runtime_error("initial velocity id is out of range");
      }

      const Point3& point = app.space.mesh().node(in);
      const Real    s     = axialCoordinate(point, axis, bounds);
      const Real    peak =
          amplitude * base_peak * sin(pi * static_cast<Real>(m + 1) * s);
      out(app.lyt.init_vel_offset + i, col) =
          comp >= 0 && comp < 3
              ? fem::velocityComponent(prof, point, peak, comp)
              : 0.0;
    }
  }
}

DenseMatrix makePerturbations(const AppNsEnVar& app,
                              const Params&     prm,
                              Index             boundary_modes,
                              Index             initial_modes,
                              Real              amplitude)
{
  const Index cols = boundary_modes + initial_modes;
  if (cols <= 0)
  {
    throw runtime_error(
        "at least one of --modes or --initial-modes must be positive");
  }

  DenseMatrix out(app.lyt.ntot, cols);
  fillBoundaryPulsePerturbations(
      out, app, prm, 0, boundary_modes, amplitude);
  fillInitialVelocityPerturbations(
      out, app, prm, boundary_modes, initial_modes, amplitude);
  return out;
}

string outputPerturbationsFile(const AppOptions& opts,
                               const Params&     prm)
{
  if (!opts.output_file.empty())
  {
    return opts.output_file;
  }
  if (!prm.inv.ens.perturbations_file.empty())
  {
    return prm.inv.ens.perturbations_file;
  }
  return "perturbations.txt";
}

string outputMeanFile(const AppOptions& opts,
                      const Params&     prm)
{
  if (!opts.mean_output_file.empty())
  {
    return opts.mean_output_file;
  }
  return prm.inv.ens.mean_file;
}

void printSummary(const AppNsEnVar& app,
                  Index             boundary_modes,
                  Index             initial_modes,
                  Real              amplitude,
                  const string&     perturbations_file,
                  const string&     mean_file)
{
  cout << "make-ensemble\n";
  cout << "  physical parameters: " << app.lyt.ntot << '\n';
  cout << "  coefficients: " << boundary_modes + initial_modes << '\n';
  cout << "  boundary modes: " << boundary_modes << '\n';
  cout << "  initial velocity modes: " << initial_modes << '\n';
  cout << "  control levels: " << app.lyt.nctr << '\n';
  cout << "  control dofs: " << app.ctr.numDofs() << '\n';
  cout << "  relative perturbation amplitude: " << amplitude << '\n';
  if (app.lyt.hasInitialVelocity())
  {
    cout << "  initial velocity parameters: " << app.lyt.niv << '\n';
  }
  cout << "  perturbations: " << perturbations_file << '\n';
  if (!mean_file.empty())
  {
    cout << "  mean: " << mean_file << '\n';
  }
}

} // namespace

int main(int argc, char** argv)
{
  int exit_code = 0;
  try
  {
    const AppOptions opts = parseAppOptions(argc, argv);
    if (opts.help)
    {
      printUsage(cout);
      return 0;
    }
    if (opts.config_file.empty())
    {
      throw runtime_error("--config FILE is required");
    }

    const Params prm = loadConfig(opts.config_file);
    AppNsEnVar   app(prm);
    const Index  boundary_modes =
        opts.modes ? *opts.modes : defaultBoundaryModeCount(app.lyt);
    const Index initial_modes =
        opts.initial_modes ? *opts.initial_modes
                           : defaultInitialModeCount(app);

    DenseMatrix perturbations = makePerturbations(
        app, prm, boundary_modes, initial_modes, opts.amplitude);
    Vector<Real> mean = app.prm0;
    const string perturbations_file =
        outputPerturbationsFile(opts, prm);
    const string mean_file = outputMeanFile(opts, prm);

    inv::writeMatrix(perturbations_file, perturbations);
    if (!mean_file.empty())
    {
      inv::writeVector(mean_file, mean);
    }

    printSummary(app,
                 boundary_modes,
                 initial_modes,
                 opts.amplitude,
                 perturbations_file,
                 mean_file);
  }
  catch (const exception& e)
  {
    cerr << FEMX_MAKE_ENSEMBLE_APP_NAME << " failed: " << e.what()
         << '\n';
    exit_code = 1;
  }

  return exit_code;
}
