#include "PoissonOpt.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/assembly/Assembler.hpp>
#include <femx/assembly/DirichletControlResidual.hpp>
#include <femx/assembly/ElementKernel.hpp>
#include <femx/assembly/FEMResidual.hpp>
#include <femx/common/Math.hpp>
#include <femx/fem/DirichletControl.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/ElementValues.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/GaussQuadrature.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/ObservationGrid.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/io/VtuWriter.hpp>
#include <femx/linalg/CsrPattern.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/opt/TaoOptimizer.hpp>
#include <femx/inverse/LeastSquaresObjective.hpp>
#include <femx/state/Linearization.hpp>
#include <femx/inverse/Objective.hpp>
#include <femx/state/Residual.hpp>
#include <femx/inverse/SumObjective.hpp>
#include <femx/runtime/PETScRuntime.hpp>
#include <femx/state/LinearStateSolver.hpp>
#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/state/StateSolver.hpp>

using namespace femx;
using namespace femx::assembly;
using namespace femx::linalg;
using namespace femx::state;
using namespace femx::inverse;
using namespace std;

#ifndef FEMX_POISSON_OPT_DEFAULT_OUTPUT_DIR
#define FEMX_POISSON_OPT_DEFAULT_OUTPUT_DIR "output"
#endif

namespace femx::examples::poisson_opt
{
namespace
{

constexpr Real boundary_eps = 1.0e-12;

class PoissonElementKernel final : public ElementKernel
{
public:
  PoissonElementKernel(const FESpace&         space,
                       const GaussQuadrature& quad)
    : space_(&space),
      quad_(&quad)
  {
  }

  void res(Index               ie,
           const Vector<Real>& u,
           const Vector<Real>& m,
           Vector<Real>&       out) const override
  {
    (void) m;
    DenseMatrix K;
    stiffness(ie, K);
    resizeOrZero(out, u.size());
    for (Index i = 0; i < K.rows(); ++i)
    {
      for (Index j = 0; j < K.cols(); ++j)
      {
        out[i] += K(i, j) * u[j];
      }
    }
  }

  void stateJac(Index               ie,
                const Vector<Real>& u,
                const Vector<Real>& m,
                DenseMatrix&        out) const override
  {
    (void) u;
    (void) m;
    stiffness(ie, out);
  }

  void paramJac(Index               ie,
                const Vector<Real>& u,
                const Vector<Real>& m,
                DenseMatrix&        out) const override
  {
    (void) ie;
    out.resize(u.size(), m.size());
    out.setZero();
  }

private:
  void stiffness(Index ie, DenseMatrix& out) const
  {
    ElementValues ev(space_->finiteElement(), *quad_);
    ev.reinit(space_->mesh().elem(ie));

    out.resize(space_->numDofsPerElem(), space_->numDofsPerElem());
    for (Index iq = 0; iq < ev.numQuadraturePoints(); ++iq)
    {
      const auto dNdx = ev.dNdx(iq);
      const Real JxW  = ev.JxW(iq);

      for (Index i = 0; i < out.rows(); ++i)
      {
        for (Index j = 0; j < out.cols(); ++j)
        {
          for (Index d = 0; d < ev.dim(); ++d)
          {
            out(i, j) += dNdx(i, d) * dNdx(j, d) * JxW;
          }
        }
      }
    }
  }

private:
  const FESpace*         space_{nullptr};
  const GaussQuadrature* quad_{nullptr};
};

Mesh makePoissonMesh(const Options& opts)
{
  if (opts.num_x_cells <= 0 || opts.num_y_cells <= 0)
  {
    throw runtime_error("Poisson optimization mesh dimensions must be positive");
  }
  if (opts.alpha < 0.0 || !isfinite(opts.alpha))
  {
    throw runtime_error("Poisson optimization alpha must be nonnegative");
  }
  if (opts.max_its <= 0)
  {
    throw runtime_error("Poisson optimization max iterations must be positive");
  }
  if (opts.obs_stride < 0)
  {
    throw runtime_error("Poisson optimization observation stride must be nonnegative");
  }
  return Mesh::makeStructuredQuad(opts.num_x_cells, opts.num_y_cells);
}

Index readIndexOption(int& i, int argc, char** argv, const string& name)
{
  if (i + 1 >= argc)
  {
    throw runtime_error(name + " requires a value");
  }

  const long value = stol(argv[++i]);
  if (value <= 0 || value > numeric_limits<Index>::max())
  {
    throw runtime_error(name + " must be a positive integer");
  }
  return static_cast<Index>(value);
}

Real readRealOption(int& i, int argc, char** argv, const string& name)
{
  if (i + 1 >= argc)
  {
    throw runtime_error(name + " requires a value");
  }

  const Real value = stod(argv[++i]);
  if (!isfinite(value))
  {
    throw runtime_error(name + " must be finite");
  }
  return value;
}

Index readNonnegativeIndexOption(int&          i,
                                 int           argc,
                                 char**        argv,
                                 const string& name)
{
  if (i + 1 >= argc)
  {
    throw runtime_error(name + " requires a value");
  }

  const long value = stol(argv[++i]);
  if (value < 0 || value > numeric_limits<Index>::max())
  {
    throw runtime_error(name + " must be a nonnegative integer");
  }
  return static_cast<Index>(value);
}

string readStringOption(int& i, int argc, char** argv, const string& name)
{
  if (i + 1 >= argc)
  {
    throw runtime_error(name + " requires a value");
  }
  return argv[++i];
}

bool readIndexAssignment(const string& arg,
                         const string& name,
                         Index&        out)
{
  const string prefix = name + "=";
  if (arg.rfind(prefix, 0) != 0)
  {
    return false;
  }

  const long value = stol(arg.substr(prefix.size()));
  if (value <= 0 || value > numeric_limits<Index>::max())
  {
    throw runtime_error(name + " must be a positive integer");
  }
  out = static_cast<Index>(value);
  return true;
}

bool readNonnegativeIndexAssignment(const string& arg,
                                    const string& name,
                                    Index&        out)
{
  const string prefix = name + "=";
  if (arg.rfind(prefix, 0) != 0)
  {
    return false;
  }

  const long value = stol(arg.substr(prefix.size()));
  if (value < 0 || value > numeric_limits<Index>::max())
  {
    throw runtime_error(name + " must be a nonnegative integer");
  }
  out = static_cast<Index>(value);
  return true;
}

bool readRealAssignment(const string& arg,
                        const string& name,
                        Real&         out)
{
  const string prefix = name + "=";
  if (arg.rfind(prefix, 0) != 0)
  {
    return false;
  }

  out = stod(arg.substr(prefix.size()));
  if (!isfinite(out))
  {
    throw runtime_error(name + " must be finite");
  }
  return true;
}

bool readStringAssignment(const string& arg,
                          const string& name,
                          string&       out)
{
  const string prefix = name + "=";
  if (arg.rfind(prefix, 0) != 0)
  {
    return false;
  }
  out = arg.substr(prefix.size());
  if (out.empty())
  {
    throw runtime_error(name + " must not be empty");
  }
  return true;
}

string lowerAscii(string value)
{
  transform(value.begin(),
            value.end(),
            value.begin(),
            [](unsigned char ch)
            { return static_cast<char>(tolower(ch)); });
  return value;
}

WorkspaceType parseWorkspaceType(const string& value)
{
  const string backend = lowerAscii(value);
  if (backend == "cpu")
  {
    return WorkspaceType::Cpu;
  }
  if (backend == "cuda" || backend == "gpu")
  {
    return WorkspaceType::Cuda;
  }
  throw runtime_error("Backend must be 'cpu' or 'cuda'");
}

WorkspaceType readBackendOption(int&          i,
                                int           argc,
                                char**        argv,
                                const string& name)
{
  return parseWorkspaceType(readStringOption(i, argc, argv, name));
}

bool readBackendAssignment(const string&  arg,
                           const string&  name,
                           WorkspaceType& out)
{
  string value;
  if (!readStringAssignment(arg, name, value))
  {
    return false;
  }
  out = parseWorkspaceType(value);
  return true;
}

bool parseOutputValue(const string& value)
{
  const string output = lowerAscii(value);
  if (output == "yes" || output == "true" || output == "on" || output == "1")
  {
    return true;
  }
  if (output == "no" || output == "false" || output == "off" || output == "0")
  {
    return false;
  }
  throw runtime_error("--output expects 'yes' or 'no'");
}

filesystem::path vtuPathFromBase(const string& output_base)
{
  filesystem::path path(output_base);
  if (path.extension() == ".vtu")
  {
    return path;
  }
  path += ".vtu";
  return path;
}

} // namespace

PoissonOptProblem::PoissonOptProblem(const Options& opts)
  : opts_(opts),
    mesh_(makePoissonMesh(opts)),
    space_(&mesh_, &fe_),
    quad_(GaussQuadrature::make(fe_.referenceElement(), 2))
{
  space_.setup();
  state_pattern_ = make_unique<CsrPattern>(assembly::makeCsrPattern(space_));
  initializeBoundaryDofs();
  initializeTrueControl();
  initializeObservationLayout();
  initializeResidual();
}

const Options& PoissonOptProblem::options() const noexcept
{
  return opts_;
}

const CsrPattern& PoissonOptProblem::statePattern() const
{
  return *state_pattern_;
}

const Residual& PoissonOptProblem::residual() const
{
  return *residual_;
}

const Objective& PoissonOptProblem::objective() const
{
  if (!obj_)
  {
    throw runtime_error("Poisson optimization objective is not prepared");
  }
  return *obj_;
}

Index PoissonOptProblem::numNodes() const noexcept
{
  return mesh_.numNodes();
}

Index PoissonOptProblem::numStates() const noexcept
{
  return space_.numDofs();
}

Index PoissonOptProblem::numParams() const noexcept
{
  return control_dofs_.size();
}

Index PoissonOptProblem::numObservations() const noexcept
{
  return obs_dofs_.size();
}

Report PoissonOptProblem::report(const Vector<Real>& prm,
                                 const Vector<Real>& state,
                                 Real                value,
                                 const Vector<Real>& grad) const
{
  if (state.size() != numStates() || prm.size() != numParams()
      || grad.size() != numParams())
  {
    throw runtime_error("Poisson report vector size mismatch");
  }

  Report out;
  out.value     = value;
  out.grad_norm = norm(grad);

  Real state_err2 = 0.0;
  for (Index i = 0; i < state.size(); ++i)
  {
    const Real err     = state[i] - target_state_[i];
    state_err2        += err * err;
    out.state_max_err  = max(out.state_max_err, abs(err));
  }
  out.state_rms_error = sqrt(state_err2 / static_cast<Real>(state.size()));

  Real control_err2 = 0.0;
  for (Index i = 0; i < prm.size(); ++i)
  {
    const Real err         = prm[i] - target_ctr_[i];
    control_err2          += err * err;
    out.control_max_error  = max(out.control_max_error, abs(err));
  }
  out.ctr_rms_error = prm.empty() ? 0.0 : sqrt(control_err2 / static_cast<Real>(prm.size()));

  return out;
}

void PoissonOptProblem::writeSolution(const Vector<Real>& prm,
                                      const Vector<Real>& state,
                                      const string&       output_base) const
{
  if (output_base.empty())
  {
    return;
  }
  if (state.size() != numStates() || prm.size() != numParams())
  {
    throw runtime_error("Poisson optimization visualization vector size mismatch");
  }
  if (target_state_.size() != numStates() || target_ctr_.size() != numParams())
  {
    throw runtime_error("Poisson optimization target vectors are not ready");
  }

  const filesystem::path output_path = vtuPathFromBase(output_base);
  if (output_path.has_parent_path())
  {
    filesystem::create_directories(output_path.parent_path());
  }

  Vector<Real> state_field(mesh_.numNodes());
  Vector<Real> target_state_field(mesh_.numNodes());
  Vector<Real> state_error(mesh_.numNodes());
  for (Index in = 0; in < mesh_.numNodes(); ++in)
  {
    const Index dof        = space_.globalDof(in, 0);
    state_field[in]        = state[dof];
    target_state_field[in] = target_state_[dof];
    state_error[in]        = state_field[in] - target_state_field[in];
  }

  Vector<Real> control(mesh_.numNodes(), 0.0);
  Vector<Real> target_control(mesh_.numNodes(), 0.0);
  Vector<Real> control_error(mesh_.numNodes(), 0.0);
  Vector<Real> control_mask(mesh_.numNodes(), 0.0);
  for (Index i = 0; i < control_dofs_.size(); ++i)
  {
    const Index node = control_dofs_[i];
    if (node < 0 || node >= mesh_.numNodes())
    {
      throw runtime_error("Poisson optimization control dof is not a mesh node");
    }
    control[node]        = prm[i];
    target_control[node] = target_ctr_[i];
    control_error[node]  = prm[i] - target_ctr_[i];
    control_mask[node]   = 1.0;
  }

  VtuWriter out;
  out.writePointData(output_path.string(),
                     mesh_,
                     {{"state", 1, &state_field},
                      {"target_state", 1, &target_state_field},
                      {"state_error", 1, &state_error},
                      {"control", 1, &control},
                      {"target_control", 1, &target_control},
                      {"control_error", 1, &control_error},
                      {"control_mask", 1, &control_mask}});
}

Real PoissonOptProblem::exactValue(const Mesh::Node& p)
{
  return sin(constants::PI * p[0]) * sinh(constants::PI * p[1])
         / sinh(constants::PI);
}

void PoissonOptProblem::initializeBoundaryDofs()
{
  set<Index> control;
  set<Index> fixed;

  for (Index in = 0; in < mesh_.numNodes(); ++in)
  {
    const Mesh::Node& point = mesh_.node(in);
    const Index       dof   = space_.globalDof(in, 0);
    if (isControlNode(point))
    {
      control.insert(dof);
    }
    else if (isBoundaryNode(point))
    {
      fixed.insert(dof);
    }
  }

  if (control.empty())
  {
    throw runtime_error("Poisson optimization found no control dofs");
  }

  for (Index dof : control)
  {
    control_dofs_.push_back(dof);
  }
  for (Index dof : fixed)
  {
    fixed_dofs_.push_back(dof);
  }

  const Real dx = 1.0 / static_cast<Real>(opts_.num_x_cells);
  control_weights_.resize(control_dofs_.size());
  for (Index i = 0; i < control_weights_.size(); ++i)
  {
    control_weights_[i] = dx;
  }
}

void PoissonOptProblem::initializeTrueControl()
{
  target_ctr_.resize(numParams());
  for (Index i = 0; i < control_dofs_.size(); ++i)
  {
    target_ctr_[i] = exactValue(mesh_.node(control_dofs_[i]));
  }
}

void PoissonOptProblem::initializeObservationLayout()
{
  const Index stride  = effectiveObservationStride();
  const Index nx      = opts_.num_x_cells;
  const Index ny      = opts_.num_y_cells;
  const Index count_x = (nx - 1) / stride;
  const Index count_y = (ny - 1) / stride;
  if (count_x <= 0 || count_y <= 0)
  {
    throw runtime_error("Poisson optimization found no observation points");
  }

  const Real dx = 1.0 / static_cast<Real>(nx);
  const Real dy = 1.0 / static_cast<Real>(ny);
  obs_points_   = fem::observationGridPoints(
      Point3{stride * dx, stride * dy, 0.0},
      {count_x, count_y, 1},
      Point3{stride * dx, stride * dy, 1.0});

  obs_dofs_.reserve(obs_points_.size());
  for (Index j = 0; j < count_y; ++j)
  {
    const Index node_y = stride * (j + 1);
    for (Index i = 0; i < count_x; ++i)
    {
      const Index node_x = stride * (i + 1);
      const Index node   = node_y * (nx + 1) + node_x;
      obs_dofs_.push_back(space_.globalDof(node, 0));
    }
  }
}

void PoissonOptProblem::initializeResidual()
{
  residual_kernel_ = make_unique<PoissonElementKernel>(space_, quad_);
  base_residual_   = make_unique<FEMResidual>(DofLayout(space_), *residual_kernel_);

  Vector<Real> fixed_values(fixed_dofs_.size(), 0.0);
  residual_ = make_unique<DirichletControlResidual>(
      *base_residual_,
      DirichletControl(control_dofs_),
      fixed_dofs_,
      0,
      numParams(),
      fixed_values);
}

Index PoissonOptProblem::effectiveObservationStride() const
{
  if (opts_.obs_stride > 0)
  {
    return opts_.obs_stride;
  }
  return max<Index>(1, min(opts_.num_x_cells, opts_.num_y_cells) / 8);
}

Vector<Real> PoissonOptProblem::observationWeights() const
{
  if (obs_dofs_.empty())
  {
    throw runtime_error("Poisson optimization has no observation dofs");
  }

  Vector<Real> weights(numStates(), 0.0);
  const Real   weight = 1.0 / static_cast<Real>(obs_dofs_.size());
  for (Index dof : obs_dofs_)
  {
    weights[dof] = weight;
  }
  return weights;
}

void PoissonOptProblem::prepareObjective(state::StateSolver& state_solver)
{
  if (obj_)
  {
    return;
  }

  state_solver.solve(target_ctr_, target_state_);

  Vector<Real> zero_ctr(numParams(), 0.0);
  Vector<Real> reg_weights(numParams(), 0.0);
  for (Index i = 0; i < reg_weights.size(); ++i)
  {
    reg_weights[i] = opts_.alpha * control_weights_[i];
  }

  tracking_obj_ = make_unique<LeastSquaresObjective>(numStates(), numParams());
  tracking_obj_->setStateTerm(target_state_, observationWeights());

  reg_ = make_unique<LeastSquaresObjective>(numStates(), numParams());
  reg_->setParamTerm(std::move(zero_ctr), std::move(reg_weights));

  obj_ = make_unique<SumObjective>(numStates(), numParams());
  obj_->add(*tracking_obj_);
  obj_->add(*reg_);
}

bool PoissonOptProblem::isBoundaryNode(const Mesh::Node& p) const
{
  return abs(p[0]) < boundary_eps || abs(p[0] - 1.0) < boundary_eps
         || abs(p[1]) < boundary_eps
         || abs(p[1] - 1.0) < boundary_eps;
}

bool PoissonOptProblem::isControlNode(const Mesh::Node& p) const
{
  return abs(p[1] - 1.0) < boundary_eps
         && p[0] > boundary_eps
         && p[0] < 1.0 - boundary_eps;
}

Result solve(PoissonOptProblem&      problem,
             state::Linearization& lin,
             linalg::LinearSolver&   fwd_lin_solver,
             linalg::LinearSolver&   adj_lin_solver)
{
  state::LinearStateSolver state_solver(problem.residual(),
                                        lin,
                                        fwd_lin_solver);

  problem.prepareObjective(state_solver);

  inverse::ReducedFunctional fn(problem.residual(),
                              problem.objective(),
                              state_solver,
                              lin,
                              adj_lin_solver);

  opt::TaoOptimizer tao(fn, PETSC_COMM_SELF);
  tao.opts().max_its = problem.options().max_its;

  opt::TaoResult     result;
  const Vector<Real> init_ctr(problem.numParams(), 0.0);
  runtime::checkPetsc(tao.solve(init_ctr, result), "TaoOptimizer::solve");

  Vector<Real> state;
  state_solver.solve(result.prm, state);

  Result out;
  out.report     = problem.report(result.prm, state, result.value, result.grad);
  out.prm        = result.prm;
  out.state      = std::move(state);
  out.tao_itr    = result.its;
  out.tao_reason = static_cast<int>(result.reason);
  out.converged  = result.converged();
  return out;
}

Options parseOptions(int    argc,
                     char** argv,
                     bool   ignore_unknown)
{
  Options opts;

  for (int i = 1; i < argc; ++i)
  {
    const string arg = argv[i];
    if (arg == "--help" || arg == "-h")
    {
      continue;
    }
    if (arg == "--nx")
    {
      opts.num_x_cells = readIndexOption(i, argc, argv, arg);
      continue;
    }
    if (arg == "--ny")
    {
      opts.num_y_cells = readIndexOption(i, argc, argv, arg);
      continue;
    }
    if (arg == "--output")
    {
      if (i + 1 < argc && string(argv[i + 1]).rfind("-", 0) != 0)
      {
        opts.write_output = parseOutputValue(argv[++i]);
      }
      else
      {
        opts.write_output = true;
      }
      continue;
    }
    if (arg == "--backend" || arg == "-b")
    {
      opts.backend = readBackendOption(i, argc, argv, arg);
      continue;
    }
    if (arg == "--alpha")
    {
      opts.alpha = readRealOption(i, argc, argv, arg);
      continue;
    }
    if (arg == "--obs-stride")
    {
      opts.obs_stride = readNonnegativeIndexOption(i, argc, argv, arg);
      continue;
    }
    if (arg == "--max-its")
    {
      opts.max_its = readIndexOption(i, argc, argv, arg);
      continue;
    }
    if (arg == "--cells" || arg.rfind("--cells=", 0) == 0)
    {
      throw runtime_error("Use --nx and --ny instead of --cells");
    }
    if (readIndexAssignment(arg, "--nx", opts.num_x_cells)
        || readIndexAssignment(arg, "--ny", opts.num_y_cells)
        || readIndexAssignment(arg, "--max-its", opts.max_its)
        || readRealAssignment(arg, "--alpha", opts.alpha))
    {
      continue;
    }
    if (readNonnegativeIndexAssignment(arg, "--obs-stride", opts.obs_stride))
    {
      continue;
    }
    if (arg.rfind("--output=", 0) == 0)
    {
      opts.write_output = parseOutputValue(arg.substr(string("--output=").size()));
      continue;
    }
    if (arg == "-o" || arg.rfind("-o=", 0) == 0)
    {
      throw runtime_error("Use --output yes or --output no");
    }
    if (readBackendAssignment(arg, "--backend", opts.backend)
        || readBackendAssignment(arg, "-b", opts.backend))
    {
      continue;
    }
    if (!ignore_unknown)
    {
      throw runtime_error("Unknown option: " + arg);
    }
  }

  if (opts.alpha < 0.0 || !isfinite(opts.alpha))
  {
    throw runtime_error("--alpha must be nonnegative");
  }
  if (opts.obs_stride < 0)
  {
    throw runtime_error("--obs-stride must be nonnegative");
  }
  return opts;
}

bool hasPoissonOptHelp(int argc, char** argv)
{
  for (int i = 1; i < argc; ++i)
  {
    const string arg = argv[i];
    if (arg == "--help" || arg == "-h")
    {
      return true;
    }
  }
  return false;
}

void printPoissonOptUsage(ostream&    out,
                          const char* app_name,
                          bool        petsc_options)
{
  out << "Usage: " << app_name
      << " [--nx N] [--ny N] [-b cpu|cuda] [--output yes|no]"
      << " [--alpha A] [--obs-stride N] [--max-its N]";
  if (petsc_options)
  {
    out << " [PETSc/TAO options]";
  }
  out << '\n';
  out << "  -b, --backend cpu|cuda selects the linear solver backend";
  if (petsc_options)
  {
    out << " (PETSc supports cpu only)";
  }
  out << '\n';
  out << "  --output yes writes a VTU file under "
      << defaultOutputDirectory()
      << '\n';
}

const char* defaultOutputDirectory()
{
  return FEMX_POISSON_OPT_DEFAULT_OUTPUT_DIR;
}

string outputStem(const Options& opts)
{
  return string("poisson-opt-nx")
         + to_string(opts.num_x_cells)
         + "-ny"
         + to_string(opts.num_y_cells);
}

void printReport(ostream&                 out,
                 const string&            backend,
                 const PoissonOptProblem& problem,
                 const Report&            report,
                 Index                    tao_itr,
                 int                      tao_reason)
{
  const Options& opts = problem.options();
  out << "Poisson optimal control (" << backend << ")\n";
  out << "  cells: " << opts.num_x_cells << " x " << opts.num_y_cells
      << '\n';
  out << "  nodes: " << problem.numNodes() << '\n';
  out << "  states: " << problem.numStates() << '\n';
  out << "  controls: " << problem.numParams() << '\n';
  out << "  observations: " << problem.numObservations() << '\n';
  out << "  observation stride: ";
  if (opts.obs_stride > 0)
  {
    out << opts.obs_stride;
  }
  else
  {
    out << "auto";
  }
  out << '\n';
  out << "  alpha: " << opts.alpha << '\n';
  out << "  tao iterations: " << tao_itr << '\n';
  out << "  tao reason: " << tao_reason << '\n';
  out << "  final value: " << report.value << '\n';
  out << "  gradient norm: " << report.grad_norm << '\n';
  out << "  state rms error: " << report.state_rms_error << '\n';
  out << "  state max error: " << report.state_max_err << '\n';
  out << "  control rms error: " << report.ctr_rms_error << '\n';
  out << "  control max error: " << report.control_max_error << '\n';
}

} // namespace femx::examples::poisson_opt
