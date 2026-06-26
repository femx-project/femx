#include <petscksp.h>
#include <petsctao.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "Helper.hpp"
#include <PetscCommon.hpp>
#include <femx/linalg/petsc/KspLinearSolver.hpp>
#include <femx/linalg/petsc/PETScMatrixOperator.hpp>
#include <femx/linalg/petsc/PETScVectorBuilder.hpp>
#include <femx/opt/TaoOptimizer.hpp>
#include <femx/state/TimeLinearStateSolver.hpp>
#include <femx/state/TimeReducedFunctional.hpp>
#include <femx/state/TimeTrajectory.hpp>

using namespace std;
using namespace femx;
using namespace femx::state;
using namespace femx::linalg;
using namespace femx::opt;
namespace inv = femx::inverse;

#ifndef FEMX_NAVIER_VAR_NEW_APP_NAME
#define FEMX_NAVIER_VAR_NEW_APP_NAME "ns-var"
#endif

namespace
{

using namespace navier_var_new;

struct AppOptions
{
  string          config_file;
  optional<Index> steps;
  bool            help = false;
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
    if (key == "--steps")
    {
      opts.steps = static_cast<Index>(
          stoi(inv::requireValue(argc, argv, i, key)));
      if (*opts.steps <= 0)
      {
        throw runtime_error("--steps must be positive");
      }
      continue;
    }
  }
  return opts;
}

void printUsage(ostream& out)
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

  void phase(const string& name)
  {
    cout << "  " << name << '\n';
  }

  void timeStep(Index step, Index total)
  {
    cout << "\r    time step " << setw(4) << step << " / "
         << setw(4) << total << flush;
    if (step >= total)
    {
      cout << '\n';
    }
  }

  void reducedStep(const char* phase_name, Index step, Index total)
  {
    const string event(phase_name);
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
      cout << "\r    adjoint step " << setw(4) << step
           << " / " << setw(4) << total << flush;
      if (step >= total)
      {
        cout << '\n';
      }
    }
  }

  void optStep(const TaoIterationInfo&       info,
               const Vector<Real>&           prm,
               const InverseParameterLayout& lyt)
  {
    cout << "  optimization step " << info.its << " / "
         << max_opt_its_ << ", objective = " << info.value
         << ", |grad| = " << info.grad_norm;
    if (info.grad.size() == lyt.ntot)
    {
      const Real ctr_norm =
          blockNorm(info.grad, lyt.coff, lyt.csz);
      if (lyt.hasInitialVelocity())
      {
        const Real init_norm =
            blockNorm(info.grad, lyt.init_vel_offset, lyt.niv);
        cout << ", |grad_u0| = " << init_norm
             << ", |grad_bc| = " << ctr_norm
             << ", u0/bc = " << ratio(init_norm, ctr_norm);
      }
      else
      {
        cout << ", |grad_bc| = " << ctr_norm;
      }
    }
    if (prm.size() == lyt.ntot
        && has_prev_prm_
        && prev_prm_.size() == lyt.ntot)
    {
      const Real ctr_step =
          blockDiffNorm(prm, prev_prm_, lyt.coff, lyt.csz);
      if (lyt.hasInitialVelocity())
      {
        const Real init_step =
            blockDiffNorm(
                prm, prev_prm_, lyt.init_vel_offset, lyt.niv);
        cout << ", |step_u0| = " << init_step
             << ", |step_bc| = " << ctr_step
             << ", step_u0/bc = " << ratio(init_step, ctr_step);
      }
      else
      {
        cout << ", |step_bc| = " << ctr_step;
      }
    }
    if (prm.size() == lyt.ntot)
    {
      prev_prm_     = prm;
      has_prev_prm_ = true;
    }
    cout << '\n';
  }

private:
  static Real blockNorm(const Vector<Real>& x, Index offset, Index size)
  {
    Real norm2 = 0.0;
    for (Index i = 0; i < size; ++i)
    {
      const Real value  = x[offset + i];
      norm2            += value * value;
    }
    return sqrt(norm2);
  }

  static Real blockDiffNorm(const Vector<Real>& x,
                            const Vector<Real>& y,
                            Index               offset,
                            Index               size)
  {
    Real norm2 = 0.0;
    for (Index i = 0; i < size; ++i)
    {
      const Real value  = x[offset + i] - y[offset + i];
      norm2            += value * value;
    }
    return sqrt(norm2);
  }

  static Real ratio(Real numerator, Real denominator)
  {
    return denominator > 0.0 ? numerator / denominator : PETSC_INFINITY;
  }

  Index        max_opt_its_{0};
  Vector<Real> prev_prm_;
  bool         has_prev_prm_{false};
};

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

void configOptimizer(
    TaoOptimizer&          optimizer,
    const OptimizerParams& opts)
{
  optimizer.opts().type     = TAOLMVM;
  optimizer.opts().abs_tol  = opts.abs_tol;
  optimizer.opts().rel_tol  = opts.rel_tol;
  optimizer.opts().step_tol = opts.step_tol;
  optimizer.opts().max_its  = opts.max_iterations;
}

void printFinalSummary(const TaoResult&      result,
                       const TimeTrajectory& tr)
{
  cout << "\nFinal summary\n";
  cout << "  parameters: " << result.prm.size() << '\n';
  cout << "  TAO converged: " << (result.converged() ? "yes" : "no")
       << ", reason = " << result.reason
       << ", iterations = " << result.its << '\n';
  cout << "  final objective: " << result.value
       << ", |grad| = " << sqrt(result.grad_norm_squared) << '\n';
  cout << "  trajectory levels: " << tr.numLevels() << '\n';
}

int run(Params& prm)
{
  PetscMPIInt rank = 0;
  PetscMPIInt size = 1;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  checkInverseRunParams(prm);

  ProgressPrinter prog(prm.inv.opt.max_iterations);

  if (rank == 0)
  {
    prog.phase("mesh and space");
  }

  AppNsVar             app(prm);
  const inv::ElemRange elems =
      inv::mpiElemRange(app.space.mesh().numElems());
  app.fem.setElemRange(elems.begin, elems.end);

  if (rank == 0)
  {
    prog.phase("linear solvers");
  }

  PETScVectorBuilder mat_row(PETSC_COMM_WORLD);
  mat_row.resize(app.space.numDofs());

  PETScMatrixOperator J_fwd_next(PETSC_COMM_WORLD);
  J_fwd_next.resize(app.pettern, mat_row);

  KspLinearSolver fwd_solver(PETSC_COMM_WORLD);
  inv::setKspOptions(fwd_solver, prm.fwd.solver);

  TimeLinearStateSolver state_solver(app.problem, J_fwd_next, fwd_solver);
  state_solver.setInitialState(app.x0);

  if (app.lyt.hasInitialVelocity())
  {
    if (rank == 0)
    {
      prog.phase("initial guess");
    }

    initializeOptGuess(app.space,
                       app.ctr,
                       prm,
                       app.lyt,
                       app.init_vdofs,
                       state_solver,
                       app.ctr_times,
                       app.prm0,
                       &app.x0);
    state_solver.resetTiming();
  }
  state_solver.setStepMonitor(
      [&prog, rank](Index step, Index total)
      {
        if (rank != 0)
        {
          return;
        }
        prog.timeStep(step, total);
      });
  unique_ptr<InitialVelocityStateSolver> init_state_solver;

  TimeStateSolver* reduced_state_solver = &state_solver;
  if (app.lyt.hasInitialVelocity())
  {
    init_state_solver =
        make_unique<InitialVelocityStateSolver>(
            state_solver,
            app.init_vdofs,
            app.lyt,
            std::move(app.x0));
    reduced_state_solver = init_state_solver.get();
  }

  if (rank == 0)
  {
    prog.phase("observation and objective");
  }

  Objective obj(prm, app);

  PETScMatrixOperator J_adj_next(PETSC_COMM_WORLD);
  PETScMatrixOperator J_adj_hist(PETSC_COMM_WORLD);
  J_adj_next.resize(app.pettern, mat_row);
  J_adj_hist.resize(app.pettern, mat_row);

  KspLinearSolver adj_solver(PETSC_COMM_WORLD);
  inv::setKspOptions(adj_solver, prm.fwd.solver);
  problem::TimeLinearization adj_lin;

  TimeReducedFunctional reduced(*reduced_state_solver,
                                app.problem,
                                adj_lin,
                                J_adj_next,
                                J_adj_hist,
                                adj_solver,
                                obj.obj);
  reduced.setProgress(
      [&prog, rank](const char* phase, Index step, Index total)
      {
        if (rank != 0)
        {
          return;
        }
        prog.reducedStep(phase, step, total);
      });
  if (app.lyt.hasInitialVelocity())
  {
    reduced.setInitialStateParamJacT(
        [lyt   = app.lyt,
         vdofs = app.init_vdofs](
            const Vector<Real>&,
            const Vector<Real>& state_grad,
            Vector<Real>&       out)
        {
          applyInitialVelocityParamJacT(
              vdofs, lyt, state_grad, out);
        });
  }

  if (rank == 0)
  {
    prog.phase("optimization");
  }

  TaoOptimizer optimizer(reduced, PETSC_COMM_WORLD);
  configOptimizer(optimizer, prm.inv.opt);
  optimizer.setVariableScale(optimizerScale(app.lyt, prm.inv.opt.scale));

  optimizer.setMonitor(
      [&prog, &app, rank](
          const TaoIterationInfo& info,
          const Vector<Real>&     current_prm)
      {
        if (rank != 0)
        {
          return;
        }
        prog.optStep(info, current_prm, app.lyt);
      });

  Vector<Real> lower;
  Vector<Real> upper;
  inverseBounds(app.space, app.ctr, prm, app.lyt, app.steps, lower, upper);
  optimizer.setBounds(lower, upper);

  TaoResult            result;
  const PetscErrorCode ierr = optimizer.solve(app.prm0, result);
  inv::checkPetsc(ierr, "TAO solve");

  if (rank == 0)
  {
    prog.phase("final forward solve");
  }

  TimeTrajectory tr;
  reduced_state_solver->solve(result.prm, tr);
  if (rank == 0)
  {
    prog.phase("write visualization");

    const Vector<Real> ctr_prm = controlParams(app.lyt, result.prm);
    writeResultViz(app.mesh,
                   app.space,
                   app.ctr,
                   tr,
                   ctr_prm,
                   app.ctr_time_stencils,
                   app.dt,
                   {prm.fwd.output.base},
                   0.0);

    printFinalSummary(result, tr);
    cout << "  visualization: " << prm.fwd.output.base << ".xdmf\n";
  }

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
  inv::setSerialOpenMp();

  int exit_code = 0;
  try
  {
    const AppOptions opts = parseAppOptions(argc, argv);
    if (opts.help)
    {
      printUsage(cout);
    }
    else
    {
      if (opts.config_file.empty())
      {
        throw runtime_error("--config FILE is required");
      }
      Params prm = loadConfig(opts.config_file);
      if (opts.steps)
      {
        prm.fwd.time.steps = *opts.steps;
      }
      exit_code = run(prm);
    }
  }
  catch (const exception& e)
  {
    cerr << FEMX_NAVIER_VAR_NEW_APP_NAME << " failed: " << e.what()
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
