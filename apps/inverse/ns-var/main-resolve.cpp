#include <petsctao.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include "Config.hpp"
#include "DirichletControl.hpp"
#include "DirichletControlEquation.hpp"
#include "NavierStokesEquation.hpp"
#include "RunSupport.hpp"
#include <femx/assembly/SparsityPatternBuilder.hpp>
#include <femx/common/Math.hpp>
#include <femx/common/Types.hpp>
#include <femx/common/Workspace.hpp>
#include <femx/eq/TimeMatrixLinearStateSolver.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/inverse/SumTimeObjectiveFunctional.hpp>
#include <femx/inverse/TimeLeastSquaresObjective.hpp>
#include <femx/inverse/TimeObservationData.hpp>
#include <femx/inverse/TimeReducedFunctional.hpp>
#include <femx/inverse/TimeRegularization.hpp>
#include <femx/inverse/petsc/TaoOptimizer.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/GmshReader.hpp>
#include <femx/mesh/Mesh.hpp>
#include <femx/system/native/SparseSystemMatrix.hpp>
#include <femx/system/resolve/ReSolveLinearSolver.hpp>

#ifndef FEMX_NAVIER_VAR_APP_NAME
#define FEMX_NAVIER_VAR_APP_NAME "ns-var"
#endif

namespace
{

using namespace femx;
using namespace femx::assembly;
using namespace femx::eq;
using namespace femx::inverse;
using namespace femx::navier_var;
using namespace femx::system;

constexpr Index kQuadOrder = 2;

void checkPetsc(PetscErrorCode ierr, const std::string& action)
{
  if (ierr != PETSC_SUCCESS)
  {
    throw std::runtime_error(
        action + " failed with PETSc error code " + std::to_string(ierr));
  }
}

VizOptions readVizOptions(const OutputParams& output)
{
  VizOptions opts;
  opts.basename = output.basename;
  return opts;
}

Index taoMaxIts(Index default_value)
{
  PetscInt value = static_cast<PetscInt>(default_value);
  checkPetsc(PetscOptionsGetInt(
                 nullptr, nullptr, "-tao_max_it", &value, nullptr),
             "PetscOptionsGetInt(-tao_max_it)");
  if (value < 0)
  {
    throw std::runtime_error("-tao_max_it must be non-negative");
  }
  return static_cast<Index>(value);
}

class ProgressPrinter
{
public:
  ProgressPrinter(Index max_opt_its, bool enabled = true)
    : max_opt_its_(max_opt_its),
      enabled_(enabled)
  {
  }

  void beginPhase(const std::string& name)
  {
    if (!enabled_)
    {
      return;
    }
    finishLine();
    current_phase_ = name;
    std::cout << "  " << current_phase_ << '\n';
  }

  void timeStep(Index step, Index total)
  {
    showStep("time step", step, total);
  }

  void reducedStep(const char* phase, Index step, Index total)
  {
    if (!enabled_)
    {
      return;
    }

    const std::string event(phase);
    if (event == "forward-begin")
    {
      ++forward_solves_;
      beginPhase("forward solve " + std::to_string(forward_solves_));
    }
    else if (event == "forward-end")
    {
      finishLine();
    }
    else if (event == "adjoint-begin")
    {
      ++adj_solves_;
      beginPhase("adjoint solve " + std::to_string(adj_solves_));
    }
    else if (event == "adjoint-step")
    {
      showStep("adjoint step", step, total);
    }
    else if (event == "adjoint-end")
    {
      finishLine();
    }
  }

  void optStep(const TaoIterationInfo& info)
  {
    if (!enabled_)
    {
      return;
    }
    finishLine();
    std::cout << "  optimization step " << info.its << " / "
              << max_opt_its_ << ", objective = " << info.value
              << ", |grad| = " << info.grad_norm << '\n';
  }

  void finishLine()
  {
    if (line_active_)
    {
      std::cout << '\n';
      line_active_ = false;
    }
  }

private:
  void showStep(const char* label, Index step, Index total)
  {
    if (!enabled_)
    {
      return;
    }
    std::cout << "\r    " << label << " " << std::setw(4) << step
              << " / " << std::setw(4) << total << std::flush;
    line_active_ = true;
    if (step >= total)
    {
      finishLine();
    }
  }

private:
  Index       max_opt_its_{0};
  bool        enabled_{true};
  bool        line_active_{false};
  Index       forward_solves_{0};
  Index       adj_solves_{0};
  std::string current_phase_;
};

void requireReSolve(const SolverParams& solver)
{
  if (solver.type == "petsc")
  {
    throw std::runtime_error(
        std::string(FEMX_NAVIER_VAR_APP_NAME)
        + " requires simulation.solver.type='auto' or 'resolve'");
  }
}

ReSolveOptions makeReSolveOptions()
{
  ReSolveOptions opts;
  opts.factor   = "none";
  opts.refactor = "none";
  opts.ir       = "none";
  opts.max_its  = 5000;
  opts.restart  = 200;
  opts.rtol     = 1.0e-8;
  opts.solve    = "fgmres";
  opts.precond  = "ilu0";
  opts.flexible = true;
  return opts;
}

struct AppOptions
{
  std::string          config_file;
  std::optional<Index> steps;
  bool                 help = false;
};

AppOptions parseAppOptions(int argc, char** argv)
{
  AppOptions options;

  const auto requireValue = [argc, argv](int& i, const std::string& key)
  {
    if (i + 1 >= argc)
    {
      throw std::runtime_error("Missing value for " + key);
    }
    return std::string(argv[++i]);
  };

  for (int i = 1; i < argc; ++i)
  {
    const std::string key(argv[i]);
    if (key == "-h" || key == "--help")
    {
      options.help = true;
      return options;
    }
    if (key == "--config" || key == "-config")
    {
      options.config_file = requireValue(i, key);
      continue;
    }
    if (key == "--steps")
    {
      options.steps = static_cast<Index>(
          std::stoi(requireValue(i, key)));
      if (*options.steps <= 0)
      {
        throw std::runtime_error("--steps must be positive");
      }
      continue;
    }
  }

  return options;
}

void printUsage(std::ostream& out)
{
  out << "Usage: " << FEMX_NAVIER_VAR_APP_NAME
      << " --config FILE [--steps N] [PETSc options]\n";
}

WorkspaceType workspaceType(const SolverParams& solver)
{
  if (solver.backend == "cuda")
  {
    return WorkspaceType::Cuda;
  }
  return WorkspaceType::Cpu;
}

int run(const Params& prm)
{
  const BCsParams&    control_bc = controlBoundary(prm);
  const Index         steps  = prm.forward.time.steps;
  const Real          dt     = prm.forward.time.dt;

  const Index max_opt_its = taoMaxIts(prm.inverse.opt.max_iterations);
  requireReSolve(prm.forward.solver);
  const auto work = workspaceType(prm.forward.solver);

  const VizOptions viz_opts = readVizOptions(prm.forward.output);

  Mesh mesh = GmshReader::read(prm.forward.mesh.file);

  auto         elem  = makeElement(mesh);
  MixedFESpace space = makeSpace(mesh, *elem);

  TimeNavierStokesParameters ns_prm;
  ns_prm.steps      = steps;
  ns_prm.dt         = dt;
  ns_prm.fluid      = fluidParams(prm);
  ns_prm.quad_order = kQuadOrder;

  NavierStokesEquation ns(space, ns_prm);

  const auto       selector = bcSelector(control_bc);
  DirichletControl ctr      = makeVelocityControl(space, selector);
  const Vector<Index> initial_velocity_dofs = initialVelocityDofs(space);
  const InverseParameterLayout param_layout =
      inverseParameterLayout(
          space, ctr, prm.inverse.initial_velocity, steps);

  const Vector<Index>      fixed_dofs = fixedDofs(space, prm, ctr);
  DirichletControlEquation eq(ns,
                              ctr,
                              fixed_dofs,
                              param_layout.control_offset,
                              param_layout.total_size);

  const CsrPattern pattern = SparsityPatternBuilder::build(space);

  SparseSystemMatrix  next_jac(pattern);
  ReSolveLinearSolver lin_solver(work, makeReSolveOptions());

  TimeMatrixLinearStateSolver state_solver(eq, next_jac, lin_solver);

  Vector<Real> x_init(eq.numStates());
  x_init.setZero();
  state_solver.setInitialState(x_init);
  std::optional<InitialVelocityStateSolver> initial_state_solver;
  TimeStateSolver* reduced_state_solver = &state_solver;
  if (param_layout.hasInitialVelocity())
  {
    initial_state_solver.emplace(
        state_solver, initial_velocity_dofs, param_layout);
    reduced_state_solver = &*initial_state_solver;
  }

  ProgressPrinter prog(max_opt_its);
  state_solver.setStepMonitor(
      [&prog](Index step, Index total)
      {
        prog.timeStep(step, total);
      });

  prog.beginPhase("read observation data");
  const TimeObservationData obs_data =
      readTimeObsData(prm.inverse.obs.file);

  auto obs = makeObsFromData(space,
                             obs_data,
                             steps,
                             eq.numStates(),
                             eq.numParams());

  TimeLeastSquaresObjective misfit(
      *obs,
      obs_data,
      misfitW(steps,
              prm.inverse.alpha,
              param_layout.hasInitialVelocity()),
      dt);

  TimeRegularization reg(
      steps,
      eq.numStates(),
      steps,
      ctr.numDofs(),
      prm.inverse.reg.time,
      prm.inverse.reg.l2);
  ParameterSliceTimeObjective control_reg(
      reg, eq.numParams(), param_layout.control_offset);
  InitialVelocityRegularization initial_velocity_reg(
      steps,
      eq.numStates(),
      param_layout,
      prm.inverse.initial_velocity.l2);

  SumTimeObjectiveFunctional obj(steps, eq.numStates(), eq.numParams());
  obj.add(misfit).add(control_reg);
  if (param_layout.hasInitialVelocity()
      && prm.inverse.initial_velocity.l2 > 0.0)
  {
    obj.add(initial_velocity_reg);
  }

  SparseSystemMatrix  adj_next_jac(pattern);
  SparseSystemMatrix  adj_prev_jac(pattern);
  ReSolveLinearSolver adj_solver(work, makeReSolveOptions());

  TimeReducedFunctional reduced(
      *reduced_state_solver,
      eq,
      adj_next_jac,
      adj_prev_jac,
      adj_solver,
      obj);

  reduced.setProgress(
      [&prog](const char* phase, Index step, Index total)
      {
        prog.reducedStep(phase, step, total);
      });
  if (param_layout.hasInitialVelocity())
  {
    reduced.setInitialStateParamJacT(
        [&initial_velocity_dofs, param_layout](
            const Vector<Real>&,
            const Vector<Real>& state_grad,
            Vector<Real>&       out)
        {
          applyInitialVelocityParamJacT(
              initial_velocity_dofs, param_layout, state_grad, out);
        });
  }

  Vector<Real> prm_init =
      initialInverseParams(space, ctr, prm, param_layout, steps, dt);

  TaoOptimizer opt(reduced);
  opt.options().type                = TAOLMVM;
  opt.options().grad_abs_tolerance  = prm.inverse.opt.grad_abs_tolerance;
  opt.options().grad_rel_tolerance  = prm.inverse.opt.grad_rel_tolerance;
  opt.options().grad_step_tolerance = prm.inverse.opt.grad_step_tolerance;
  opt.options().max_its             = max_opt_its;
  opt.options().use_opts_db         = prm.inverse.opt.use_options_database;

  opt.setMonitor(
      [&prog](const TaoIterationInfo& info, const Vector<Real>&)
      {
        prog.optStep(info);
      });

  Vector<Real> lower;
  Vector<Real> upper;
  inverseBounds(space, ctr, prm, param_layout, steps, lower, upper);
  opt.setBounds(lower, upper);

  TaoResult result;
  prog.beginPhase("optimization");
  const PetscErrorCode ierr = opt.solve(prm_init, result);
  if (ierr != PETSC_SUCCESS)
  {
    throw std::runtime_error(
        "TAO solve failed with PETSc error code " + std::to_string(ierr));
  }

  const Real  fwd_assembly = state_solver.assemblySeconds();
  const Real  fwd_solve    = state_solver.solveSeconds();
  const Index fwd_assembly_calls = state_solver.assemblyCalls();
  const Index fwd_solve_calls    = state_solver.solveCalls();
  const Real  adj_assembly       = reduced.assemblySeconds();
  const Real  adj_solve          = reduced.solveSeconds();
  const Index adj_assembly_calls = reduced.assemblyCalls();
  const Index adj_solve_calls    = reduced.solveCalls();

  prog.beginPhase("write output");
  TimeStateTrajectory opt_tr;
  reduced_state_solver->solve(result.prm, opt_tr);
  writeForwardViz(mesh, space, opt_tr, dt, viz_opts);

  std::cout << "\nFinal summary\n";
  std::cout << "  observation file: " << prm.inverse.obs.file << '\n';
  std::cout << "  reg.time: " << prm.inverse.reg.time
            << ", reg.l2: " << prm.inverse.reg.l2 << '\n';
  std::cout << "  TAO converged: " << (result.converged() ? "yes" : "no")
            << ", reason = " << result.reason
            << ", iterations = " << result.its << '\n';
  std::cout << "  final objective: " << result.value
            << ", |grad| = " << std::sqrt(result.grad_norm_squared)
            << '\n';
  std::cout << "  TIMING assembly_s=" << (fwd_assembly + adj_assembly)
            << " solve_s=" << (fwd_solve + adj_solve)
            << " forward_assembly_s=" << fwd_assembly
            << " forward_solve_s=" << fwd_solve
            << " forward_assembly_calls=" << fwd_assembly_calls
            << " forward_solve_calls=" << fwd_solve_calls
            << " adjoint_assembly_s=" << adj_assembly
            << " adjoint_solve_s=" << adj_solve
            << " adjoint_assembly_calls=" << adj_assembly_calls
            << " adjoint_solve_calls=" << adj_solve_calls
            << '\n';
  std::cout << "  output: " << viz_opts.basename << '\n';

  return result.converged() ? 0 : 2;
}

} // namespace

int main(int argc, char** argv)
{
  PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
  if (ierr != PETSC_SUCCESS)
  {
    return 1;
  }

  int exit_code = 0;
  try
  {
    const AppOptions options = parseAppOptions(argc, argv);
    if (options.help)
    {
      printUsage(std::cout);
      exit_code = 0;
    }
    else
    {
      if (options.config_file.empty())
      {
        throw std::runtime_error("--config FILE is required");
      }
      Params prm = loadConfig(options.config_file);
      if (options.steps)
      {
        prm.forward.time.steps = *options.steps;
      }
      exit_code = run(prm);
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << FEMX_NAVIER_VAR_APP_NAME << " failed: " << e.what() << '\n';
    exit_code = 1;
  }

  ierr = PetscFinalize();
  if (ierr != PETSC_SUCCESS && exit_code == 0)
  {
    return 1;
  }
  return exit_code;
}
