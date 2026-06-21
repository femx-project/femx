#include <petsctao.h>

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "Helper.hpp"
#include <femx/common/Workspace.hpp>
#include <femx/linalg/native/SparseMatrixOperator.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/opt/TaoOptimizer.hpp>
#include <femx/solve/TimeLinearStateSolver.hpp>
#include <femx/solve/TimeReducedFunctional.hpp>
#include <femx/solve/TimeTrajectory.hpp>

#ifndef FEMX_NAVIER_VAR_NEW_APP_NAME
#define FEMX_NAVIER_VAR_NEW_APP_NAME "ns-var-resolve"
#endif

namespace
{

using namespace femx;
using namespace femx::linalg;
using namespace femx::navier_var_new;
using namespace femx::opt;
using namespace femx::solve;

struct AppOptions
{
  std::string          config_file;
  std::optional<Index> steps;
  bool                 help = false;
};

void checkPetsc(PetscErrorCode ierr, const std::string& action)
{
  if (ierr != PETSC_SUCCESS)
  {
    throw std::runtime_error(
        action + " failed with PETSc error code " + std::to_string(ierr));
  }
}

std::string requireValue(int                argc,
                         char**             argv,
                         int&               i,
                         const std::string& key)
{
  if (i + 1 >= argc)
  {
    throw std::runtime_error("Missing value for " + key);
  }
  return std::string(argv[++i]);
}

AppOptions parseAppOptions(int argc, char** argv)
{
  AppOptions options;
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
      options.config_file = requireValue(argc, argv, i, key);
      continue;
    }
    if (key == "--steps")
    {
      options.steps = static_cast<Index>(
          std::stoi(requireValue(argc, argv, i, key)));
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
  out << "Usage: " << FEMX_NAVIER_VAR_NEW_APP_NAME
      << " --config FILE [--steps N] [PETSc options]\n";
}

class ProgressPrinter
{
public:
  explicit ProgressPrinter(Index max_opt_its)
    : max_opt_its_(max_opt_its)
  {
  }

  void phase(const std::string& name)
  {
    std::cout << "  " << name << '\n';
  }

  void timeStep(Index step, Index total)
  {
    std::cout << "\r    time step " << std::setw(4) << step << " / "
              << std::setw(4) << total << std::flush;
    if (step >= total)
    {
      std::cout << '\n';
    }
  }

  void reducedStep(const char* phase_name, Index step, Index total)
  {
    const std::string event(phase_name);
    if (event == "forward-begin")
    {
      phase("forward solve");
    }
    else if (event == "adjoint-begin")
    {
      phase("adjoint solve");
    }
    else if (event == "adjoint-step")
    {
      std::cout << "\r    adjoint step " << std::setw(4) << step
                << " / " << std::setw(4) << total << std::flush;
      if (step >= total)
      {
        std::cout << '\n';
      }
    }
  }

  void optStep(const TaoIterationInfo&        info,
               const Vector<Real>&            prm,
               const InverseParameterLayout&  layout)
  {
    std::cout << "  optimization step " << info.its << " / "
              << max_opt_its_ << ", objective = " << info.value
              << ", |grad| = " << info.grad_norm;
    if (info.grad.size() == layout.total_size)
    {
      const Real ctr_norm =
          blockNorm(info.grad, layout.ctr_offset, layout.ctr_size);
      if (layout.hasInitialVelocity())
      {
        const Real init_norm =
            blockNorm(info.grad, layout.init_vel_offset, layout.init_vel_size);
        std::cout << ", |grad_u0| = " << init_norm
                  << ", |grad_bc| = " << ctr_norm
                  << ", u0/bc = " << ratio(init_norm, ctr_norm);
      }
      else
      {
        std::cout << ", |grad_bc| = " << ctr_norm;
      }
    }
    if (prm.size() == layout.total_size
        && has_prev_prm_
        && prev_prm_.size() == layout.total_size)
    {
      const Real ctr_step =
          blockDiffNorm(prm, prev_prm_, layout.ctr_offset, layout.ctr_size);
      if (layout.hasInitialVelocity())
      {
        const Real init_step =
            blockDiffNorm(
                prm, prev_prm_, layout.init_vel_offset, layout.init_vel_size);
        std::cout << ", |step_u0| = " << init_step
                  << ", |step_bc| = " << ctr_step
                  << ", step_u0/bc = " << ratio(init_step, ctr_step);
      }
      else
      {
        std::cout << ", |step_bc| = " << ctr_step;
      }
    }
    if (prm.size() == layout.total_size)
    {
      prev_prm_     = prm;
      has_prev_prm_ = true;
    }
    std::cout << '\n';
  }

private:
  static Real blockNorm(const Vector<Real>& x, Index offset, Index size)
  {
    Real norm2 = 0.0;
    for (Index i = 0; i < size; ++i)
    {
      const Real value = x[offset + i];
      norm2 += value * value;
    }
    return std::sqrt(norm2);
  }

  static Real blockDiffNorm(const Vector<Real>& x,
                            const Vector<Real>& y,
                            Index               offset,
                            Index               size)
  {
    Real norm2 = 0.0;
    for (Index i = 0; i < size; ++i)
    {
      const Real value = x[offset + i] - y[offset + i];
      norm2 += value * value;
    }
    return std::sqrt(norm2);
  }

  static Real ratio(Real numerator, Real denominator)
  {
    return denominator > 0.0 ? numerator / denominator : PETSC_INFINITY;
  }

  Index        max_opt_its_{0};
  Vector<Real> prev_prm_;
  bool         has_prev_prm_{false};
};

void requireReSolve(const SolverParams& solver)
{
  if (solver.type == "petsc")
  {
    throw std::runtime_error(
        std::string(FEMX_NAVIER_VAR_NEW_APP_NAME)
        + " requires simulation.solver.type='auto' or 'resolve'");
  }
}

WorkspaceType workspaceType(const SolverParams& solver)
{
  if (solver.backend == "cuda")
  {
    return WorkspaceType::Cuda;
  }
  return WorkspaceType::Cpu;
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

Vector<Real> unboundedLower(Index size)
{
  Vector<Real> out(size);
  for (Index i = 0; i < size; ++i)
  {
    out[i] = -PETSC_INFINITY;
  }
  return out;
}

Vector<Real> unboundedUpper(Index size)
{
  Vector<Real> out(size);
  for (Index i = 0; i < size; ++i)
  {
    out[i] = PETSC_INFINITY;
  }
  return out;
}

void configureOptimizer(TaoOptimizer& optimizer, const OptimizerParams& options)
{
  optimizer.options().type     = TAOLMVM;
  optimizer.options().abs_tol  = options.abs_tol;
  optimizer.options().rel_tol  = options.rel_tol;
  optimizer.options().step_tol = options.step_tol;
  optimizer.options().max_its  = options.max_iterations;
}

void printFinalSummary(const TaoResult& result, const TimeTrajectory& trajectory)
{
  std::cout << "\nFinal summary\n";
  std::cout << "  parameters: " << result.prm.size() << '\n';
  std::cout << "  TAO converged: " << (result.converged() ? "yes" : "no")
            << ", reason = " << result.reason
            << ", iterations = " << result.its << '\n';
  std::cout << "  final objective: " << result.value
            << ", |grad| = " << std::sqrt(result.grad_norm_squared) << '\n';
  std::cout << "  trajectory levels: " << trajectory.numLevels() << '\n';
}

bool isRootRank()
{
  PetscMPIInt rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  return rank == 0;
}

int run(Params& prm)
{
  checkInverseRunParams(prm);
  requireReSolve(prm.fwd.solver);
  const WorkspaceType work = workspaceType(prm.fwd.solver);

  ProgressPrinter prog(prm.inv.opt.max_iterations);

  prog.phase("mesh and space");

  AppNsVar app(prm);

  prog.phase("linear solvers");

  SparseMatrixOperator fwd_next_jac(app.pattern);
  ReSolveLinearSolver  fwd_solver(work, makeReSolveOptions());

  TimeLinearStateSolver state_solver(app.eq, fwd_next_jac, fwd_solver);
  state_solver.setInitialState(app.x0);
  Vector<Real> x0 = app.x0;
  if (app.layout.hasInitialVelocity())
  {
    prog.phase("initial guess");

    initializeOptimizationGuess(app.space,
                                app.ctr,
                                prm,
                                app.layout,
                                app.init_vdofs,
                                state_solver,
                                app.ctr_times,
                                app.prm0,
                                &x0);
    state_solver.resetTiming();
  }
  state_solver.setStepMonitor(
      [&prog](Index step, Index total)
      {
        prog.timeStep(step, total);
      });
  std::unique_ptr<InitialVelocityStateSolver> initial_state_solver;
  TimeStateSolver*                            reduced_state_solver =
      &state_solver;
  if (app.layout.hasInitialVelocity())
  {
    initial_state_solver =
        std::make_unique<InitialVelocityStateSolver>(
            state_solver,
            app.init_vdofs,
            app.layout,
            x0);
    reduced_state_solver = initial_state_solver.get();
  }

  prog.phase("observation and objective");

  Objective objective(prm, app);

  SparseMatrixOperator adj_next_jac(app.pattern);
  SparseMatrixOperator adj_prev_jac(app.pattern);
  ReSolveLinearSolver  adj_solver(work, makeReSolveOptions());

  TimeReducedFunctional reduced(*reduced_state_solver,
                                app.eq,
                                adj_next_jac,
                                adj_prev_jac,
                                adj_solver,
                                objective.obj);
  reduced.setProgress(
      [&prog](const char* phase, Index step, Index total)
      {
        prog.reducedStep(phase, step, total);
      });
  if (app.layout.hasInitialVelocity())
  {
    reduced.setInitialStateParamJacT(
        [layout        = app.layout,
         velocity_dofs = app.init_vdofs](
            const Vector<Real>&,
            const Vector<Real>& state_grad,
            Vector<Real>&       out)
        {
          applyInitialVelocityParamJacT(
              velocity_dofs, layout, state_grad, out);
        });
  }

  prog.phase("optimization");

  TaoOptimizer optimizer(reduced, PETSC_COMM_SELF);
  configureOptimizer(optimizer, prm.inv.opt);
  optimizer.setVariableScale(optimizerScale(app.layout, prm.inv.opt.scale));
  optimizer.setMonitor(
      [&prog, &app](
          const TaoIterationInfo& info,
          const Vector<Real>&     current_prm)
      {
        prog.optStep(info, current_prm, app.layout);
      });
  Vector<Real> lower;
  Vector<Real> upper;
  inverseBounds(app.space, app.ctr, prm, app.layout, app.steps, lower, upper);
  optimizer.setBounds(lower, upper);

  TaoResult            result;
  const PetscErrorCode ierr = optimizer.solve(app.prm0, result);
  checkPetsc(ierr, "TAO solve");

  prog.phase("final forward solve");

  TimeTrajectory tr;
  reduced_state_solver->solve(result.prm, tr);

  prog.phase("write visualization");

  const Vector<Real> ctr_prm = controlParams(app.layout, result.prm);
  writeResultViz(app.mesh,
                 app.space,
                 app.ctr,
                 tr,
                 ctr_prm,
                 app.ctr_time_stencils,
                 app.dt,
                 {prm.fwd.output.basename},
                 0.0);

  printFinalSummary(result, tr);
  std::cout << "  visualization: " << prm.fwd.output.basename << ".xdmf\n";

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
    if (!isRootRank())
    {
      ierr = PetscFinalize();
      return ierr == PETSC_SUCCESS ? 0 : 1;
    }

    const AppOptions options = parseAppOptions(argc, argv);
    if (options.help)
    {
      printUsage(std::cout);
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
        prm.fwd.time.steps = *options.steps;
      }
      exit_code = run(prm);
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << FEMX_NAVIER_VAR_NEW_APP_NAME << " failed: " << e.what()
              << '\n';
    exit_code = 1;
  }

  ierr = PetscFinalize();
  if (ierr != PETSC_SUCCESS && exit_code == 0)
  {
    return 1;
  }
  return exit_code;
}
