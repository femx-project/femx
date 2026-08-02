#include "PoissonOptProblem.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "../ExampleHelper.hpp"
#include <femx/ad/Enzyme.hpp>
#include <femx/common/Math.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/ObservationGrid.hpp>
#include <femx/io/VtuWriter.hpp>
#include <femx/runtime/Cli.hpp>

using namespace femx;
using namespace femx::fem;
using namespace femx::inverse;
using namespace femx::io;

#ifndef FEMX_POISSON_OPT_DEFAULT_OUTPUT_DIR
#define FEMX_POISSON_OPT_DEFAULT_OUTPUT_DIR "output"
#endif

namespace femx::examples::poisson_opt
{
namespace
{

constexpr Real boundary_epsilon = 1.0e-12;

void validateOptions(const Options& opts)
{
  if (opts.num_x_cells <= 0 || opts.num_y_cells <= 0)
  {
    throw std::runtime_error(
        "Poisson optimization mesh dimensions must be positive");
  }
  if (opts.alpha < 0.0 || !std::isfinite(opts.alpha))
  {
    throw std::runtime_error(
        "Poisson optimization alpha must be nonnegative");
  }
  if (opts.max_itrs <= 0)
  {
    throw std::runtime_error(
        "Poisson optimization maximum iterations must be positive");
  }
  if (opts.observation_stride < 0)
  {
    throw std::runtime_error(
        "Poisson optimization observation stride must be nonnegative");
  }
}

Mesh makePoissonMesh(const Options& opts)
{
  validateOptions(opts);
  return Mesh::makeStructuredQuad(
      opts.num_x_cells, opts.num_y_cells);
}

Index readNonnegativeIndex(int&               idx,
                           int                argc,
                           char**             argv,
                           const std::string& option)
{
  const long val = std::stol(runtime::requireValue(
      argc, argv, idx, option));

  if (val < 0 || val > std::numeric_limits<Index>::max())
  {
    throw std::runtime_error(
        option + " must be a nonnegative integer");
  }

  return static_cast<Index>(val);
}

Real readReal(int&               idx,
              int                argc,
              char**             argv,
              const std::string& option)
{
  const Real val = std::stod(runtime::requireValue(
      argc, argv, idx, option));

  if (!std::isfinite(val))
  {
    throw std::runtime_error(option + " must be finite");
  }

  return val;
}

std::string lowerAscii(std::string val)
{
  std::transform(val.begin(),
                 val.end(),
                 val.begin(),
                 [](unsigned char character)
                 {
                   return static_cast<char>(
                       std::tolower(character));
                 });

  return val;
}

bool readOutputValue(const std::string& val)
{
  const std::string output = lowerAscii(val);

  if (output == "yes" || output == "no")
  {
    return parseYesNo(output, "--output");
  }
  if (output == "true" || output == "on" || output == "1")
  {
    return true;
  }
  if (output == "false" || output == "off" || output == "0")
  {
    return false;
  }

  throw std::runtime_error("--output expects 'yes' or 'no'");
}

bool readPositiveAssignment(const std::string& argument,
                            const std::string& option,
                            Index&             out)
{
  const std::string prefix = option + "=";

  if (argument.rfind(prefix, 0) != 0)
  {
    return false;
  }
  out = parsePositiveIndex(argument.substr(prefix.size()), option);
  return true;
}

bool readNonnegativeAssignment(const std::string& argument,
                               const std::string& option,
                               Index&             out)
{
  const std::string prefix = option + "=";

  if (argument.rfind(prefix, 0) != 0)
  {
    return false;
  }

  const long val = std::stol(argument.substr(prefix.size()));
  if (val < 0 || val > std::numeric_limits<Index>::max())
  {
    throw std::runtime_error(
        option + " must be a nonnegative integer");
  }
  out = static_cast<Index>(val);

  return true;
}

bool readRealAssignment(const std::string& argument,
                        const std::string& option,
                        Real&              out)
{
  const std::string prefix = option + "=";

  if (argument.rfind(prefix, 0) != 0)
  {
    return false;
  }

  out = std::stod(argument.substr(prefix.size()));
  if (!std::isfinite(out))
  {
    throw std::runtime_error(option + " must be finite");
  }

  return true;
}

bool readBackendAssignment(const std::string& argument,
                           MemorySpace&       out)
{
  constexpr const char* prefix = "--backend=";

  if (argument.rfind(prefix, 0) != 0)
  {
    return false;
  }
  const std::string val = argument.substr(std::string(prefix).size());
  out                   = parseBackend(val);

  return true;
}

std::filesystem::path vtuPathFromBase(const std::string& base)
{
  std::filesystem::path path(base);
  if (path.extension() != ".vtu")
  {
    path += ".vtu";
  }
  return path;
}

std::filesystem::path observationVtuPath(
    std::filesystem::path solution_path)
{
  solution_path.replace_filename(
      solution_path.stem().string() + ".observations.vtu");
  return solution_path;
}

} // namespace

PoissonOptProblem::PoissonOptProblem(const Options& opts)
  : opts_(opts),
    mesh_(makePoissonMesh(opts)),
    space_(&mesh_, &fe_)
{
  // Use the same Q1 Poisson discretization as the forward example.
  space_.setup();

  elem_data_ = makeElementQuadData(
      space_,
      GaussQuadrature::make(
          fe_.shape(), 2));

  // Map local element residuals and stiffness entries into global CSR rows.
  assm_map_ = assembly::makeAssemblyMap(space_.dofMap());

  // Define which boundary values are optimization parameters, manufacture a
  // target control, and select the interior state entries to observe.
  initBoundary();
  initTargetControl();
  initObservations();
}

const Options& PoissonOptProblem::opts() const noexcept
{
  return opts_;
}

const Mesh& PoissonOptProblem::mesh() const noexcept
{
  return mesh_;
}

const HostElementQuadData&
PoissonOptProblem::elementData() const noexcept
{
  return elem_data_;
}

const assembly::HostAssemblyMap&
PoissonOptProblem::assemblyMap() const noexcept
{
  return assm_map_;
}

const assembly::HostBoundaryMap&
PoissonOptProblem::boundaryMap() const noexcept
{
  return boundary_map_;
}

const HostVector<Index>&
PoissonOptProblem::controlDofs() const noexcept
{
  return ctr_dofs_;
}

const HostVector<Real>&
PoissonOptProblem::targetControl() const noexcept
{
  return target_ctr_;
}

const Objective& PoissonOptProblem::objective() const
{
  if (!obj_)
  {
    throw std::runtime_error(
        "Poisson optimization objective is not prepared");
  }
  return *obj_;
}

void PoissonOptProblem::setupObjective(HostVector<Real> target_state)
{
  if (obj_)
  {
    return;
  }
  if (target_state.size() != numStates())
  {
    throw std::runtime_error(
        "Poisson target state size mismatch");
  }
  target_state_ = std::move(target_state);

  // Minimizes J(x,m) = J_misfit(x) + J_reg(m), where
  // J_misfit(x) = 1/2 sum_i w_i (x_i - d_i)^2 and J_reg(m) = alpha/2 sum_i w_i^m m_i^2.

  HostVector<Real> reg_target(numParameters(), 0.0);
  HostVector<Real> reg_weights(numParameters(), 0.0);
  for (Index idx = 0; idx < reg_weights.size(); ++idx)
  {
    reg_weights[idx] = opts_.alpha * ctr_weights_[idx];
  }

  // J_misfit(x) term.
  misfit_ = std::make_unique<LeastSquaresObjective>(numStates(), numParameters());
  misfit_->setStateTerm(target_state_, observationWeights());

  // J_reg(m) term.
  reg_ = std::make_unique<LeastSquaresObjective>(numStates(), numParameters());
  reg_->setParamTerm(std::move(reg_target), std::move(reg_weights));

  obj_ = std::make_unique<SumObjective>(numStates(), numParameters());
  obj_->add(*misfit_);
  obj_->add(*reg_);
}

Index PoissonOptProblem::numNodes() const noexcept
{
  return mesh_.numNodes();
}

Index PoissonOptProblem::numStates() const noexcept
{
  return space_.numDofs();
}

Index PoissonOptProblem::numParameters() const noexcept
{
  return ctr_dofs_.size();
}

Index PoissonOptProblem::numObservations() const noexcept
{
  return obs_dofs_.size();
}

Report PoissonOptProblem::report(
    const HostVector<Real>& ctr,
    const HostVector<Real>& state,
    Real                    val,
    const HostVector<Real>& grad) const
{
  if (state.size() != numStates()
      || ctr.size() != numParameters()
      || grad.size() != numParameters())
  {
    throw std::runtime_error(
        "Poisson optimization report vector size mismatch");
  }

  Report out;
  out.value         = val;
  out.gradient_norm = norm(grad);

  Real state_error_squared = 0.0;
  for (Index idx = 0; idx < state.size(); ++idx)
  {
    const Real error     = state[idx] - target_state_[idx];
    state_error_squared += error * error;
    out.state_max_error  = std::max(out.state_max_error, std::abs(error));
  }
  out.state_rms_error =
      std::sqrt(state_error_squared
                / static_cast<Real>(state.size()));

  Real ctr_error_squared = 0.0;
  for (Index idx = 0; idx < ctr.size(); ++idx)
  {
    const Real error       = ctr[idx] - target_ctr_[idx];
    ctr_error_squared     += error * error;
    out.control_max_error  = std::max(out.control_max_error, std::abs(error));
  }

  out.control_rms_error =
      ctr.empty()
          ? 0.0
          : std::sqrt(ctr_error_squared
                      / static_cast<Real>(ctr.size()));

  return out;
}

void PoissonOptProblem::writeSolution(
    const HostVector<Real>& ctr,
    const HostVector<Real>& state,
    const std::string&      base) const
{
  if (base.empty())
  {
    return;
  }
  if (state.size() != numStates()
      || ctr.size() != numParameters())
  {
    throw std::runtime_error(
        "Poisson optimization visualization vector size mismatch");
  }
  if (target_state_.size() != numStates())
  {
    throw std::runtime_error(
        "Poisson optimization target state is not prepared");
  }

  const std::filesystem::path path = vtuPathFromBase(base);
  if (path.has_parent_path())
  {
    std::filesystem::create_directories(
        path.parent_path());
  }
  writeFields(ctr, state, path.string());
  writeObservations(state, path.string());
}

void PoissonOptProblem::writeFields(
    const HostVector<Real>& ctr,
    const HostVector<Real>& state,
    const std::string&      path) const
{
  HostVector<Real> state_field(mesh_.numNodes());
  HostVector<Real> target_state_field(mesh_.numNodes());
  HostVector<Real> state_error(mesh_.numNodes());

  for (Index in = 0; in < mesh_.numNodes(); ++in)
  {
    const Index dof        = space_.globalDof(in, 0);
    state_field[in]        = state[dof];
    target_state_field[in] = target_state_[dof];
    state_error[in] =
        state_field[in] - target_state_field[in];
  }

  HostVector<Real> ctr_field(mesh_.numNodes(), 0.0);
  HostVector<Real> target_ctr_field(mesh_.numNodes(), 0.0);
  HostVector<Real> ctr_error(mesh_.numNodes(), 0.0);
  HostVector<Real> ctr_mask(mesh_.numNodes(), 0.0);

  for (Index idx = 0; idx < ctr_dofs_.size(); ++idx)
  {
    const Index node = ctr_dofs_[idx];
    if (node < 0 || node >= mesh_.numNodes())
    {
      throw std::runtime_error(
          "Poisson optimization control degree of freedom is not a mesh node");
    }
    ctr_field[node]        = ctr[idx];
    target_ctr_field[node] = target_ctr_[idx];
    ctr_error[node] =
        ctr[idx] - target_ctr_[idx];
    ctr_mask[node] = 1.0;
  }

  VtuWriter writer;
  writer.writePointData(
      path,
      mesh_,
      {{"state", 1, &state_field},
       {"target_state", 1, &target_state_field},
       {"state_error", 1, &state_error},
       {"control", 1, &ctr_field},
       {"target_control", 1, &target_ctr_field},
       {"control_error", 1, &ctr_error},
       {"control_mask", 1, &ctr_mask}});
}

void PoissonOptProblem::writeObservations(
    const HostVector<Real>& state,
    const std::string&      path) const
{
  if (obs_points_.size() != obs_dofs_.size())
  {
    throw std::runtime_error(
        "Poisson optimization observation layout is inconsistent");
  }

  HostVector<Real> target_values(obs_dofs_.size(), 0.0);
  HostVector<Real> predicted_values(obs_dofs_.size(), 0.0);
  HostVector<Real> misfit_values(obs_dofs_.size(), 0.0);
  HostVector<Real> weights(obs_dofs_.size(), 0.0);

  const Real weight = 1.0 / static_cast<Real>(obs_dofs_.size());
  for (Index idx = 0; idx < obs_dofs_.size(); ++idx)
  {
    const Index dof       = obs_dofs_[idx];
    target_values[idx]    = target_state_[dof];
    predicted_values[idx] = state[dof];
    misfit_values[idx]    = state[dof] - target_state_[dof];
    weights[idx]          = weight;
  }

  VtuWriter writer;
  writer.writePointCloud(
      observationVtuPath(path).string(),
      obs_points_,
      {{"observation", 1, &target_values},
       {"prediction", 1, &predicted_values},
       {"misfit", 1, &misfit_values},
       {"weight", 1, &weights}});
}

Real PoissonOptProblem::exactValue(const Mesh::Node& p)
{
  const Real wave_number = 2.0 * constants::PI;
  return std::sin(wave_number * p[0])
         * std::sinh(wave_number * p[1])
         / std::sinh(wave_number);
}

void PoissonOptProblem::initBoundary()
{
  std::set<Index> ctr_rows;
  std::set<Index> fixed_rows;

  // Interior nodes of the top edge are controlled by m. The remaining outer
  // boundary is fixed to zero.
  for (Index in = 0; in < mesh_.numNodes(); ++in)
  {
    const Mesh::Node& point = mesh_.node(in);
    const Index       dof   = space_.globalDof(in, 0);
    if (isControlNode(point))
    {
      ctr_rows.insert(dof);
    }
    else if (isBoundaryNode(point))
    {
      fixed_rows.insert(dof);
    }
  }
  if (ctr_rows.empty())
  {
    throw std::runtime_error(
        "Poisson optimization found no control degrees of freedom");
  }

  HostVector<Index> boundary_rows;
  boundary_rows.reserve(static_cast<Index>(
      ctr_rows.size() + fixed_rows.size()));

  // Put controlled rows first so the residual can copy m directly into the
  // leading boundary values; fixed rows retain the trailing zero values.
  for (Index i : ctr_rows)
  {
    ctr_dofs_.push_back(i);
    boundary_rows.push_back(i);
  }
  for (Index i : fixed_rows)
  {
    boundary_rows.push_back(i);
  }
  boundary_map_ = assembly::makeBoundaryMap(boundary_rows);

  // Lump the one-dimensional boundary integral onto the control nodes.
  const Real cell_width = 1.0 / static_cast<Real>(opts_.num_x_cells);
  ctr_weights_.assign(ctr_dofs_.size(), cell_width);
}

void PoissonOptProblem::initTargetControl()
{
  // Sampling a harmonic manufactured solution on the controlled edge gives
  // a known control from which noise-free observations can be generated.
  target_ctr_.resize(numParameters());
  for (Index idx = 0; idx < ctr_dofs_.size();
       ++idx)
  {
    target_ctr_[idx] =
        exactValue(mesh_.node(ctr_dofs_[idx]));
  }
}

void PoissonOptProblem::initObservations()
{
  // Observe a regular subset of interior nodal values. Their target values
  // are filled later by solving the state equation at target_ctr_.
  const Index stride = effectiveObservationStride();
  const Index count_x =
      (opts_.num_x_cells - 1) / stride;
  const Index count_y =
      (opts_.num_y_cells - 1) / stride;
  if (count_x <= 0 || count_y <= 0)
  {
    throw std::runtime_error(
        "Poisson optimization found no observation points");
  }

  const Real cell_width  = 1.0 / static_cast<Real>(opts_.num_x_cells);
  const Real cell_height = 1.0 / static_cast<Real>(opts_.num_y_cells);

  obs_points_ = observationGridPoints(
      Point3{stride * cell_width,
             stride * cell_height,
             0.0},
      {count_x, count_y, 1},
      Point3{stride * cell_width,
             stride * cell_height,
             1.0});

  obs_dofs_.reserve(obs_points_.size());
  for (Index iy = 0; iy < count_y; ++iy)
  {
    const Index node_y = stride * (iy + 1);
    for (Index ix = 0; ix < count_x; ++ix)
    {
      const Index node_x = stride * (ix + 1);
      const Index node   = node_y * (opts_.num_x_cells + 1) + node_x;
      obs_dofs_.push_back(space_.globalDof(node, 0));
    }
  }
}

Index PoissonOptProblem::effectiveObservationStride() const
{
  if (opts_.observation_stride > 0)
  {
    return opts_.observation_stride;
  }
  return std::max<Index>(
      1,
      std::min(opts_.num_x_cells,
               opts_.num_y_cells)
          / 8);
}

HostVector<Real>
PoissonOptProblem::observationWeights() const
{
  if (obs_dofs_.empty())
  {
    throw std::runtime_error(
        "Poisson optimization has no observation degrees of freedom");
  }
  HostVector<Real> weights(numStates(), 0.0);
  const Real       weight = 1.0 / static_cast<Real>(obs_dofs_.size());

  for (Index dof : obs_dofs_)
  {
    weights[dof] = weight;
  }

  return weights;
}

bool PoissonOptProblem::isBoundaryNode(
    const Mesh::Node& p) const
{
  return std::abs(p[0]) < boundary_epsilon
         || std::abs(p[0] - 1.0) < boundary_epsilon
         || std::abs(p[1]) < boundary_epsilon
         || std::abs(p[1] - 1.0) < boundary_epsilon;
}

bool PoissonOptProblem::isControlNode(
    const Mesh::Node& p) const
{
  return std::abs(p[1] - 1.0) < boundary_epsilon
         && p[0] > boundary_epsilon
         && p[0] < 1.0 - boundary_epsilon;
}

Options parseOptions(int    argc,
                     char** argv,
                     bool   ignore_unknown)
{
  Options opts;

  for (int idx = 1; idx < argc; ++idx)
  {
    const std::string argument = argv[idx];
    if (argument == "--help" || argument == "-h")
    {
      continue;
    }
    if (argument == "--nx")
    {
      opts.num_x_cells = parsePositiveIndex(
          runtime::requireValue(
              argc, argv, idx, argument),
          argument);
      continue;
    }
    if (argument == "--ny")
    {
      opts.num_y_cells = parsePositiveIndex(
          runtime::requireValue(
              argc, argv, idx, argument),
          argument);
      continue;
    }
    if (argument == "--backend" || argument == "-b")
    {
      opts.memspace = parseBackend(
          runtime::requireValue(
              argc, argv, idx, argument));
      continue;
    }
    if (argument == "--output")
    {
      if (idx + 1 < argc
          && std::string(argv[idx + 1]).rfind("-", 0) != 0)
      {
        opts.write_output = readOutputValue(argv[++idx]);
      }
      else
      {
        opts.write_output = true;
      }
      continue;
    }
    if (argument == "--alpha")
    {
      opts.alpha =
          readReal(idx, argc, argv, argument);
      continue;
    }
    if (argument == "--obs-stride")
    {
      opts.observation_stride =
          readNonnegativeIndex(
              idx, argc, argv, argument);
      continue;
    }
    if (argument == "--max-its")
    {
      opts.max_itrs = parsePositiveIndex(
          runtime::requireValue(
              argc, argv, idx, argument),
          argument);
      continue;
    }
    if (argument == "--cells"
        || argument.rfind("--cells=", 0) == 0)
    {
      throw std::runtime_error(
          "Use --nx and --ny instead of --cells");
    }
    if (readPositiveAssignment(
            argument, "--nx", opts.num_x_cells)
        || readPositiveAssignment(
            argument, "--ny", opts.num_y_cells)
        || readPositiveAssignment(
            argument,
            "--max-its",
            opts.max_itrs)
        || readRealAssignment(
            argument, "--alpha", opts.alpha)
        || readNonnegativeAssignment(
            argument,
            "--obs-stride",
            opts.observation_stride)
        || readBackendAssignment(
            argument, opts.memspace))
    {
      continue;
    }
    if (argument.rfind("--output=", 0) == 0)
    {
      opts.write_output = readOutputValue(
          argument.substr(
              std::string("--output=").size()));
      continue;
    }
    if (argument == "-o"
        || argument.rfind("-o=", 0) == 0)
    {
      throw std::runtime_error(
          "Use --output yes or --output no");
    }
    if (!ignore_unknown)
    {
      throw std::runtime_error(
          "Unknown option: " + argument);
    }
  }

  validateOptions(opts);
  return opts;
}

void printUsage(std::ostream& out,
                const char*   app_name,
                bool          petsc_opts)
{
  out << "Usage: " << app_name
      << " [--nx N] [--ny N] [-b cpu|cuda]"
      << " [--output yes|no] [--alpha A]"
      << " [--obs-stride N] [--max-its N]";
  if (petsc_opts)
  {
    out << " [PETSc/TAO options]";
  }
  out << '\n';
  out << "  -b, --backend cpu|cuda selects the execution backend\n";
  out << "  --output yes writes VTU files under "
      << outputDir() << '\n';
}

const char* outputDir()
{
  return FEMX_POISSON_OPT_DEFAULT_OUTPUT_DIR;
}

std::string outputStem(const Options& opts)
{
  return std::string("poisson-opt-nx")
         + std::to_string(opts.num_x_cells)
         + "-ny" + std::to_string(opts.num_y_cells);
}

void printReport(std::ostream&            out,
                 const std::string&       config,
                 const PoissonOptProblem& problem,
                 const Report&            rep,
                 Index                    itrs,
                 int                      reason)
{
  const Options& opts = problem.opts();
  out << "Poisson optimal control (" << config << ")\n";
  out << "  backend: " << backendName(opts.memspace) << '\n';
  out << "  parameter VJP: "
      << (ad::has_enzyme ? "Enzyme" : "analytic fallback")
      << '\n';
  out << "  cells: " << opts.num_x_cells << " x "
      << opts.num_y_cells << '\n';
  out << "  nodes: " << problem.numNodes() << '\n';
  out << "  states: " << problem.numStates() << '\n';
  out << "  controls: " << problem.numParameters() << '\n';
  out << "  observations: " << problem.numObservations()
      << '\n';
  out << "  observation stride: ";
  if (opts.observation_stride > 0)
  {
    out << opts.observation_stride;
  }
  else
  {
    out << "auto";
  }
  out << '\n';
  out << "  alpha: " << opts.alpha << '\n';
  out << "  TAO iterations: " << itrs << '\n';
  out << "  TAO reason: " << reason << '\n';
  out << "  final value: " << rep.value << '\n';
  out << "  gradient norm: " << rep.gradient_norm
      << '\n';
  out << "  state RMS error: " << rep.state_rms_error
      << '\n';
  out << "  state max error: " << rep.state_max_error
      << '\n';
  out << "  control RMS error: "
      << rep.control_rms_error << '\n';
  out << "  control max error: "
      << rep.control_max_error << '\n';
}

} // namespace femx::examples::poisson_opt
