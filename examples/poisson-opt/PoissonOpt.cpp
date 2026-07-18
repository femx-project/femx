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

#include "../poisson/PoissonOperator.hpp"
#include <femx/assembly/Assembly.hpp>
#include <femx/assembly/BoundaryMap.hpp>
#include <femx/common/Math.hpp>
#include <femx/fem/DofLayout.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Geometry.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/ObservationGrid.hpp>
#include <femx/fem/elements/LagrangeQuadQ1.hpp>
#include <femx/inverse/LeastSquaresObjective.hpp>
#include <femx/inverse/Objective.hpp>
#include <femx/inverse/ReducedFunctional.hpp>
#include <femx/inverse/SumObjective.hpp>
#include <femx/io/VtuWriter.hpp>
#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/native/MapCsrMatrix.hpp>
#include <femx/opt/TaoOptimizer.hpp>
#include <femx/runtime/PETScRuntime.hpp>
#include <femx/state/LinearStateSolver.hpp>
#include <femx/state/Linearization.hpp>
#include <femx/state/Residual.hpp>
#include <femx/state/StateSolver.hpp>

using namespace femx;
using namespace femx::assembly;
using namespace femx::fem;
using namespace femx::io;
using namespace femx::linalg;
using namespace femx::state;
using namespace femx::inverse;

#ifndef FEMX_POISSON_OPT_DEFAULT_OUTPUT_DIR
#define FEMX_POISSON_OPT_DEFAULT_OUTPUT_DIR "output"
#endif

namespace femx::examples::poisson_opt
{
namespace
{

constexpr Real boundary_eps = 1.0e-12;

/**
 * @brief AssemblyMap-backed residual for the optimization workflow.
 *
 * The finite-element work is performed by Geometry/AssemblyMap/ElementView,
 * while MatrixLinearization publishes the resulting state and control
 * operators to ReducedFunctional and TAO.
 */
class PoissonMapResidual final : public Residual
{
public:
  PoissonMapResidual(const HostGeometry&    geom,
                     const HostAssemblyMap& map,
                     Array<Index>           control_dofs,
                     Array<Index>           fixed_dofs)
    : geom_(&geom),
      map_(&map),
      control_dofs_(std::move(control_dofs)),
      fixed_dofs_(std::move(fixed_dofs)),
      jac_(map.graph())
  {
    Array<Index> bc_dofs = control_dofs_;
    bc_dofs.reserve(control_dofs_.size() + fixed_dofs_.size());
    for (Index dof : fixed_dofs_)
    {
      bc_dofs.push_back(dof);
    }
    bc_map_ = makeBoundaryMap(bc_dofs, map.graph());
  }

  Dimensions dims() const override
  {
    return {map_->numStates(),
            control_dofs_.size(),
            map_->numRes()};
  }

  void res(const HostVector& state,
           const HostVector& prm,
           HostVector&       out) const override
  {
    checkVectors(state, prm);
    assembleRaw(state, out);
    replaceRes(bc_map_, state, bcVals(prm), out);
  }

  void linearize(const HostVector& state,
                 const HostVector& prm,
                 Linearization&    out) const override
  {
    checkVectors(state, prm);
    auto* mats = dynamic_cast<MatrixLinearization*>(&out);
    if (mats == nullptr)
    {
      throw std::runtime_error(
          "PoissonMapResidual requires MatrixLinearization output");
    }

    HostVector res;
    assembleRaw(state, res);
    replaceRows(bc_map_, jac_);
    copyStateJac(mats->stateMat());
    assemblePrmJac(mats->paramMat());
  }

private:
  void checkVectors(const HostVector& state,
                    const HostVector& prm) const
  {
    const Dimensions shape = dims();
    if (state.size() != shape.num_states || prm.size() != shape.num_param)
    {
      throw std::runtime_error("PoissonMapResidual vector size mismatch");
    }
  }

  void assembleRaw(const HostVector& state, HostVector& res) const
  {
    CpuContext ctx;
    assembly::assemble(poisson::PoissonQuadQ1Operator{},
                       *geom_,
                       *map_,
                       state,
                       res,
                       jac_,
                       ctx);
  }

  HostVector bcVals(const HostVector& prm) const
  {
    HostVector vals(bc_map_.numBcs(), 0.0);
    for (Index i = 0; i < control_dofs_.size(); ++i)
    {
      vals[i] = prm[i];
    }
    return vals;
  }

  void copyStateJac(MatrixOperator& out) const
  {
    if (out.numRows() != jac_.rows()
        || out.numCols() != jac_.cols())
    {
      throw std::runtime_error(
          "PoissonMapResidual state matrix size mismatch");
    }

    if (auto* csr = dynamic_cast<MapCsrMatrix*>(&out))
    {
      HostCsrMatrix& dst = csr->mat();
      if (dst.graph().layoutId() != jac_.graph().layoutId())
      {
        throw std::runtime_error(
            "PoissonMapResidual state matrix uses a different CSR graph");
      }
      std::copy(jac_.valsData(),
                jac_.valsData() + jac_.nnz(),
                dst.valsData());
    }
    else
    {
      out.setZero();
      for (Index row = 0; row < jac_.rows(); ++row)
      {
        for (Index k = jac_.rowPtrData()[row];
             k < jac_.rowPtrData()[row + 1];
             ++k)
        {
          out.set(row, jac_.colIndData()[k], jac_.valsData()[k]);
        }
      }
    }
    out.finalize();
  }

  void assemblePrmJac(MatrixOperator& out) const
  {
    out.resize(map_->numRes(), control_dofs_.size());
    out.setZero();
    for (Index col = 0; col < control_dofs_.size(); ++col)
    {
      out.set(control_dofs_[col], col, -1.0);
    }
    out.finalize();
  }

  const HostGeometry*    geom_{nullptr};
  const HostAssemblyMap* map_{nullptr};
  Array<Index>           control_dofs_;
  Array<Index>           fixed_dofs_;
  HostBoundaryMap        bc_map_;
  mutable HostCsrMatrix  jac_;
};

Mesh makePoissonMesh(const Options& opts)
{
  if (opts.num_x_cells <= 0 || opts.num_y_cells <= 0)
  {
    throw std::runtime_error("Poisson optimization mesh dimensions must be positive");
  }
  if (opts.alpha < 0.0 || !std::isfinite(opts.alpha))
  {
    throw std::runtime_error("Poisson optimization alpha must be nonnegative");
  }
  if (opts.max_its <= 0)
  {
    throw std::runtime_error("Poisson optimization max iterations must be positive");
  }
  if (opts.obs_stride < 0)
  {
    throw std::runtime_error("Poisson optimization observation stride must be nonnegative");
  }
  return Mesh::makeStructuredQuad(opts.num_x_cells, opts.num_y_cells);
}

Index readIndexOption(int& i, int argc, char** argv, const std::string& name)
{
  if (i + 1 >= argc)
  {
    throw std::runtime_error(name + " requires a value");
  }

  const long value = std::stol(argv[++i]);
  if (value <= 0 || value > std::numeric_limits<Index>::max())
  {
    throw std::runtime_error(name + " must be a positive integer");
  }
  return static_cast<Index>(value);
}

Real readRealOption(int& i, int argc, char** argv, const std::string& name)
{
  if (i + 1 >= argc)
  {
    throw std::runtime_error(name + " requires a value");
  }

  const Real value = std::stod(argv[++i]);
  if (!std::isfinite(value))
  {
    throw std::runtime_error(name + " must be finite");
  }
  return value;
}

Index readNonnegativeIndexOption(int&               i,
                                 int                argc,
                                 char**             argv,
                                 const std::string& name)
{
  if (i + 1 >= argc)
  {
    throw std::runtime_error(name + " requires a value");
  }

  const long value = std::stol(argv[++i]);
  if (value < 0 || value > std::numeric_limits<Index>::max())
  {
    throw std::runtime_error(name + " must be a nonnegative integer");
  }
  return static_cast<Index>(value);
}

bool readIndexAssignment(const std::string& arg,
                         const std::string& name,
                         Index&             out)
{
  const std::string prefix = name + "=";
  if (arg.rfind(prefix, 0) != 0)
  {
    return false;
  }

  const long value = std::stol(arg.substr(prefix.size()));
  if (value <= 0 || value > std::numeric_limits<Index>::max())
  {
    throw std::runtime_error(name + " must be a positive integer");
  }
  out = static_cast<Index>(value);
  return true;
}

bool readNonnegativeIndexAssignment(const std::string& arg,
                                    const std::string& name,
                                    Index&             out)
{
  const std::string prefix = name + "=";
  if (arg.rfind(prefix, 0) != 0)
  {
    return false;
  }

  const long value = std::stol(arg.substr(prefix.size()));
  if (value < 0 || value > std::numeric_limits<Index>::max())
  {
    throw std::runtime_error(name + " must be a nonnegative integer");
  }
  out = static_cast<Index>(value);
  return true;
}

bool readRealAssignment(const std::string& arg,
                        const std::string& name,
                        Real&              out)
{
  const std::string prefix = name + "=";
  if (arg.rfind(prefix, 0) != 0)
  {
    return false;
  }

  out = std::stod(arg.substr(prefix.size()));
  if (!std::isfinite(out))
  {
    throw std::runtime_error(name + " must be finite");
  }
  return true;
}

std::string lowerAscii(std::string value)
{
  transform(value.begin(),
            value.end(),
            value.begin(),
            [](unsigned char ch)
            { return static_cast<char>(tolower(ch)); });
  return value;
}

bool parseOutputValue(const std::string& value)
{
  const std::string output = lowerAscii(value);
  if (output == "yes" || output == "true" || output == "on" || output == "1")
  {
    return true;
  }
  if (output == "no" || output == "false" || output == "off" || output == "0")
  {
    return false;
  }
  throw std::runtime_error("--output expects 'yes' or 'no'");
}

std::filesystem::path vtuPathFromBase(const std::string& base)
{
  std::filesystem::path path(base);
  if (path.extension() == ".vtu")
  {
    return path;
  }
  path += ".vtu";
  return path;
}

std::filesystem::path observationVtuPath(std::filesystem::path solution_path)
{
  solution_path.replace_filename(solution_path.stem().string()
                                 + ".observations.vtu");
  return solution_path;
}

} // namespace

PoissonOptProblem::PoissonOptProblem(const Options& opts)
  : opts_(opts),
    mesh_(makePoissonMesh(opts)),
    space_(&mesh_, &fe_)
{
  space_.setup();
  geom_      = makeGeometry(mesh_);
  state_map_ = assembly::makeAssemblyMap(DofLayout(space_));
  initializeBoundaryDofs();
  initializeTrueControl();
  initializeObservationLayout();
  initializeResidual();
}

const Options& PoissonOptProblem::options() const noexcept
{
  return opts_;
}

const assembly::HostAssemblyMap&
PoissonOptProblem::stateMap() const
{
  return state_map_;
}

const Residual& PoissonOptProblem::residual() const
{
  return *res_;
}

const Objective& PoissonOptProblem::objective() const
{
  if (!obj_)
  {
    throw std::runtime_error("Poisson optimization objective is not prepared");
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
  return ctr_dofs_.size();
}

Index PoissonOptProblem::numObservations() const noexcept
{
  return obs_dofs_.size();
}

Report PoissonOptProblem::report(const HostVector& prm,
                                 const HostVector& state,
                                 Real              value,
                                 const HostVector& grad) const
{
  if (state.size() != numStates() || prm.size() != numParams()
      || grad.size() != numParams())
  {
    throw std::runtime_error("Poisson report vector size mismatch");
  }

  Report out;
  out.value     = value;
  out.grad_norm = norm(grad);

  Real state_err2 = 0.0;
  for (Index i = 0; i < state.size(); ++i)
  {
    const Real err     = state[i] - target_state_[i];
    state_err2        += err * err;
    out.state_max_err  = std::max(out.state_max_err, std::abs(err));
  }
  out.state_rms_err = std::sqrt(state_err2 / static_cast<Real>(state.size()));

  Real ctr_err2 = 0.0;
  for (Index i = 0; i < prm.size(); ++i)
  {
    const Real err   = prm[i] - target_ctr_[i];
    ctr_err2        += err * err;
    out.ctr_max_err  = std::max(out.ctr_max_err, std::abs(err));
  }
  out.ctr_rms_err = prm.empty() ? 0.0 : std::sqrt(ctr_err2 / static_cast<Real>(prm.size()));

  return out;
}

void PoissonOptProblem::writeSolution(const HostVector&  prm,
                                      const HostVector&  state,
                                      const std::string& base) const
{
  if (base.empty())
  {
    return;
  }
  if (state.size() != numStates() || prm.size() != numParams())
  {
    throw std::runtime_error("Poisson optimization visualization vector size mismatch");
  }
  if (target_state_.size() != numStates() || target_ctr_.size() != numParams())
  {
    throw std::runtime_error("Poisson optimization target vectors are not ready");
  }

  const std::filesystem::path path = vtuPathFromBase(base);
  if (path.has_parent_path())
  {
    std::filesystem::create_directories(path.parent_path());
  }
  writeFields(prm, state, path.string());
  writeObs(state, path.string());
}

void PoissonOptProblem::writeFields(const HostVector&  prm,
                                    const HostVector&  state,
                                    const std::string& path) const
{
  HostVector state_field(mesh_.numNodes());
  HostVector target_state_field(mesh_.numNodes());
  HostVector state_err(mesh_.numNodes());
  for (Index in = 0; in < mesh_.numNodes(); ++in)
  {
    const Index dof        = space_.globalDof(in, 0);
    state_field[in]        = state[dof];
    target_state_field[in] = target_state_[dof];
    state_err[in]          = state_field[in] - target_state_field[in];
  }

  HostVector ctr(mesh_.numNodes(), 0.0);
  HostVector target_ctr(mesh_.numNodes(), 0.0);
  HostVector ctr_err(mesh_.numNodes(), 0.0);
  HostVector ctr_mask(mesh_.numNodes(), 0.0);
  for (Index i = 0; i < ctr_dofs_.size(); ++i)
  {
    const Index node = ctr_dofs_[i];
    if (node < 0 || node >= mesh_.numNodes())
    {
      throw std::runtime_error("Poisson optimization control dof is not a mesh node");
    }
    ctr[node]        = prm[i];
    target_ctr[node] = target_ctr_[i];
    ctr_err[node]    = prm[i] - target_ctr_[i];
    ctr_mask[node]   = 1.0;
  }

  VtuWriter out;
  out.writePointData(path,
                     mesh_,
                     {{"state", 1, &state_field},
                      {"target_state", 1, &target_state_field},
                      {"state_err", 1, &state_err},
                      {"ctr", 1, &ctr},
                      {"target_ctr", 1, &target_ctr},
                      {"ctr_error", 1, &ctr_err},
                      {"ctr_mask", 1, &ctr_mask}});
}

void PoissonOptProblem::writeObs(const HostVector&  state,
                                 const std::string& path) const
{
  if (obs_points_.size() != obs_dofs_.size())
  {
    throw std::runtime_error("Poisson optimization observation layout is inconsistent");
  }

  const Index comps = space_.numComponents();
  HostVector  obs_point_value(obs_dofs_.size(), 0.0);
  HostVector  obs_point_pred(obs_dofs_.size(), 0.0);
  HostVector  obs_point_misfit(obs_dofs_.size(), 0.0);
  HostVector  obs_weight(obs_dofs_.size(), 0.0);
  const Real  weight = 1.0 / static_cast<Real>(obs_dofs_.size());
  for (Index i = 0; i < obs_dofs_.size(); ++i)
  {
    const Index dof = obs_dofs_[i];
    if (dof < 0 || dof >= numStates() || dof % comps != 0)
    {
      throw std::runtime_error("Poisson optimization observation dof is not a mesh node");
    }
    const Index node = dof / comps;
    if (node < 0 || node >= mesh_.numNodes())
    {
      throw std::runtime_error("Poisson optimization observation node is out of range");
    }

    obs_point_value[i]  = target_state_[dof];
    obs_point_pred[i]   = state[dof];
    obs_point_misfit[i] = state[dof] - target_state_[dof];
    obs_weight[i]       = weight;
  }

  VtuWriter out;
  out.writePointCloud(observationVtuPath(path).string(),
                      obs_points_,
                      {{"obs_value", 1, &obs_point_value},
                       {"obs_pred", 1, &obs_point_pred},
                       {"obs_misfit", 1, &obs_point_misfit},
                       {"obs_weight", 1, &obs_weight}});
}

Real PoissonOptProblem::exactValue(const Mesh::Node& p)
{
  const Real wave_number = 2.0 * constants::PI;
  return std::sin(wave_number * p[0]) * std::sinh(wave_number * p[1])
         / std::sinh(wave_number);
}

void PoissonOptProblem::initializeBoundaryDofs()
{
  std::set<Index> ctr;
  std::set<Index> fixed;

  for (Index in = 0; in < mesh_.numNodes(); ++in)
  {
    const Mesh::Node& point = mesh_.node(in);
    const Index       dof   = space_.globalDof(in, 0);
    if (isControlNode(point))
    {
      ctr.insert(dof);
    }
    else if (isBoundaryNode(point))
    {
      fixed.insert(dof);
    }
  }

  if (ctr.empty())
  {
    throw std::runtime_error("Poisson optimization found no control dofs");
  }

  for (Index dof : ctr)
  {
    ctr_dofs_.push_back(dof);
  }
  for (Index dof : fixed)
  {
    fixed_dofs_.push_back(dof);
  }

  const Real dx = 1.0 / static_cast<Real>(opts_.num_x_cells);
  ctr_weights_.resize(ctr_dofs_.size());
  for (Index i = 0; i < ctr_weights_.size(); ++i)
  {
    ctr_weights_[i] = dx;
  }
}

void PoissonOptProblem::initializeTrueControl()
{
  target_ctr_.resize(numParams());
  for (Index i = 0; i < ctr_dofs_.size(); ++i)
  {
    target_ctr_[i] = exactValue(mesh_.node(ctr_dofs_[i]));
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
    throw std::runtime_error("Poisson optimization found no observation points");
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
  res_ = std::make_unique<PoissonMapResidual>(geom_,
                                              state_map_,
                                              ctr_dofs_,
                                              fixed_dofs_);
}

Index PoissonOptProblem::effectiveObservationStride() const
{
  if (opts_.obs_stride > 0)
  {
    return opts_.obs_stride;
  }
  return std::max<Index>(1, std::min(opts_.num_x_cells, opts_.num_y_cells) / 8);
}

HostVector PoissonOptProblem::observationWeights() const
{
  if (obs_dofs_.empty())
  {
    throw std::runtime_error("Poisson optimization has no observation dofs");
  }

  HostVector weights(numStates(), 0.0);
  const Real weight = 1.0 / static_cast<Real>(obs_dofs_.size());
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

  HostVector zero_ctr(numParams(), 0.0);
  HostVector reg_weights(numParams(), 0.0);
  for (Index i = 0; i < reg_weights.size(); ++i)
  {
    reg_weights[i] = opts_.alpha * ctr_weights_[i];
  }

  misfit_ = std::make_unique<LeastSquaresObjective>(numStates(), numParams());
  misfit_->setStateTerm(target_state_, observationWeights());

  reg_ = std::make_unique<LeastSquaresObjective>(numStates(), numParams());
  reg_->setParamTerm(std::move(zero_ctr), std::move(reg_weights));

  obj_ = std::make_unique<SumObjective>(numStates(), numParams());
  obj_->add(*misfit_);
  obj_->add(*reg_);
}

bool PoissonOptProblem::isBoundaryNode(const Mesh::Node& p) const
{
  return std::abs(p[0]) < boundary_eps || std::abs(p[0] - 1.0) < boundary_eps
         || std::abs(p[1]) < boundary_eps
         || std::abs(p[1] - 1.0) < boundary_eps;
}

bool PoissonOptProblem::isControlNode(const Mesh::Node& p) const
{
  return std::abs(p[1] - 1.0) < boundary_eps
         && p[0] > boundary_eps
         && p[0] < 1.0 - boundary_eps;
}

Result solve(PoissonOptProblem&    problem,
             state::Linearization& lin,
             linalg::LinearSolver& fwd_lin_solver,
             linalg::LinearSolver& adj_lin_solver)
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

  opt::TaoResult   result;
  const HostVector init_ctr(problem.numParams(), 0.0);
  runtime::checkPetsc(tao.solve(init_ctr, result), "TaoOptimizer::solve");

  HostVector state;
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
    const std::string arg = argv[i];
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
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("-", 0) != 0)
      {
        opts.write_output = parseOutputValue(argv[++i]);
      }
      else
      {
        opts.write_output = true;
      }
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
      throw std::runtime_error("Use --nx and --ny instead of --cells");
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
      opts.write_output = parseOutputValue(arg.substr(std::string("--output=").size()));
      continue;
    }
    if (arg == "-o" || arg.rfind("-o=", 0) == 0)
    {
      throw std::runtime_error("Use --output yes or --output no");
    }
    if (!ignore_unknown)
    {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }

  if (opts.alpha < 0.0 || !std::isfinite(opts.alpha))
  {
    throw std::runtime_error("--alpha must be nonnegative");
  }
  if (opts.obs_stride < 0)
  {
    throw std::runtime_error("--obs-stride must be nonnegative");
  }
  return opts;
}

void printUsage(std::ostream& out,
                const char*   app_name,
                bool          petsc_options)
{
  out << "Usage: " << app_name
      << " [--nx N] [--ny N] [--output yes|no]"
      << " [--alpha A] [--obs-stride N] [--max-its N]";
  if (petsc_options)
  {
    out << " [PETSc/TAO options]";
  }
  out << '\n';
  out << "  --output yes writes VTU files under "
      << outputDir()
      << '\n';
}

const char* outputDir()
{
  return FEMX_POISSON_OPT_DEFAULT_OUTPUT_DIR;
}

std::string outputStem(const Options& opts)
{
  return std::string("poisson-opt-nx")
         + std::to_string(opts.num_x_cells)
         + "-ny"
         + std::to_string(opts.num_y_cells);
}

void printReport(std::ostream&            out,
                 const std::string&       backend,
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
  out << "  state rms error: " << report.state_rms_err << '\n';
  out << "  state max error: " << report.state_max_err << '\n';
  out << "  control rms error: " << report.ctr_rms_err << '\n';
  out << "  control max error: " << report.ctr_max_err << '\n';
}

} // namespace femx::examples::poisson_opt
