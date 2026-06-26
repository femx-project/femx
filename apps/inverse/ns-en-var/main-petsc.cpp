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
#include <femx/state/TimeTrajectory.hpp>

using namespace std;
using namespace femx;
using namespace femx::state;
using namespace femx::linalg;
using namespace femx::opt;
namespace inv = femx::inverse;

#ifndef FEMX_NAVIER_EN_VAR_APP_NAME
#define FEMX_NAVIER_EN_VAR_APP_NAME "ns-en-var"
#endif

namespace
{

using namespace navier_en_var;

struct AppOptions
{
  string          config_file;
  optional<Index> steps;
  optional<Index> max_iterations;
  optional<Real>  abs_tol;
  optional<Real>  rel_tol;
  optional<Real>  step_tol;
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
    if (key == "--max-iterations")
    {
      opts.max_iterations = static_cast<Index>(
          stoi(inv::requireValue(argc, argv, i, key)));
      if (*opts.max_iterations < 0)
      {
        throw runtime_error("--max-iterations must be nonnegative");
      }
      continue;
    }
    if (key == "--abs-tol")
    {
      opts.abs_tol = stod(inv::requireValue(argc, argv, i, key));
      if (!isfinite(*opts.abs_tol) || *opts.abs_tol < 0.0)
      {
        throw runtime_error("--abs-tol must be nonnegative");
      }
      continue;
    }
    if (key == "--rel-tol")
    {
      opts.rel_tol = stod(inv::requireValue(argc, argv, i, key));
      if (!isfinite(*opts.rel_tol) || *opts.rel_tol < 0.0)
      {
        throw runtime_error("--rel-tol must be nonnegative");
      }
      continue;
    }
    if (key == "--step-tol")
    {
      opts.step_tol = stod(inv::requireValue(argc, argv, i, key));
      if (!isfinite(*opts.step_tol) || *opts.step_tol < 0.0)
      {
        throw runtime_error("--step-tol must be nonnegative");
      }
      continue;
    }
  }
  return opts;
}

void printUsage(ostream& out)
{
  out << "Usage: " << FEMX_NAVIER_EN_VAR_APP_NAME
      << " --config FILE [--steps N] [--max-iterations N]"
      << " [--abs-tol TOL] [--rel-tol TOL] [--step-tol TOL]"
      << " [PETSc options]\n";
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

  void optStep(const TaoIterationInfo&       info,
               const Vector<Real>&           alpha,
               const Vector<Real>&           prm,
               const InverseParameterLayout& lyt)
  {
    cout << "  optimization step " << info.its << " / "
         << max_opt_its_ << ", objective = " << info.value
         << ", |grad| = " << info.grad_norm;
    if (has_prev_alpha_ && prev_alpha_.size() == alpha.size())
    {
      cout << ", |step_alpha| = "
           << diffNorm(alpha, prev_alpha_, 0, alpha.size());
    }
    if (prm.size() == lyt.ntot
        && has_prev_prm_
        && prev_prm_.size() == lyt.ntot)
    {
      const Real ctr_step =
          diffNorm(prm, prev_prm_, lyt.coff, lyt.csz);
      if (lyt.hasInitialVelocity())
      {
        const Real init_step =
            diffNorm(prm, prev_prm_, lyt.init_vel_offset, lyt.niv);
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
    prev_alpha_     = alpha;
    has_prev_alpha_ = true;
    cout << '\n';
  }

private:
  static Real diffNorm(const Vector<Real>& x,
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
  Vector<Real> prev_alpha_;
  Vector<Real> prev_prm_;
  bool         has_prev_alpha_{false};
  bool         has_prev_prm_{false};
};

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

Vector<Real> zeroCoefficients(const EnsembleBasis& basis)
{
  return Vector<Real>(basis.numCoefficients(), 0.0);
}

Vector<Real> physicalParams(const EnsembleBasis& basis,
                            const Vector<Real>&  alpha)
{
  Vector<Real> prm;
  basis.apply(alpha, prm);
  return prm;
}

class LinearObservationEnVarFunctional final
{
public:
  LinearObservationEnVarFunctional(const EnsembleBasis& obs_basis,
                                   Vector<Real>         obs,
                                   Real                 obs_weight,
                                   Real                 prior_weight)
    : obs_basis_(obs_basis),
      obs_(std::move(obs)),
      obs_weight_(obs_weight),
      prior_weight_(prior_weight)
  {
    if (obs_.size() != obs_basis_.numPhysicalParams()
        || obs_weight_ < 0.0 || prior_weight_ < 0.0)
    {
      throw runtime_error("invalid linear observation EnVar parameters");
    }
  }

  Index numParams() const
  {
    return obs_basis_.numCoefficients();
  }

  Real value(const Vector<Real>& alpha)
  {
    Vector<Real> grad;
    return valueGrad(alpha, grad);
  }

  Real valueGrad(const Vector<Real>& alpha, Vector<Real>& grad)
  {
    checkAlpha(alpha);

    Vector<Real> res;
    obs_basis_.apply(alpha, res);
    Real out = 0.0;
    for (Index i = 0; i < res.size(); ++i)
    {
      res[i] -= obs_[i];
      out    += 0.5 * obs_weight_ * res[i] * res[i];
    }

    obs_basis_.applyT(res, grad);
    for (Index j = 0; j < grad.size(); ++j)
    {
      grad[j]  = obs_weight_ * grad[j] + prior_weight_ * alpha[j];
      out     += 0.5 * prior_weight_ * alpha[j] * alpha[j];
    }
    return out;
  }

private:
  void checkAlpha(const Vector<Real>& alpha) const
  {
    if (alpha.size() != numParams())
    {
      throw runtime_error(
          "LinearObservationEnVar coefficient size mismatch");
    }
  }

private:
  const EnsembleBasis& obs_basis_;
  Vector<Real>         obs_;
  Real                 obs_weight_{1.0};
  Real                 prior_weight_{1.0};
};

Real coefficientPrior(const Vector<Real>& alpha,
                      Real                prior_weight)
{
  Real out = 0.0;
  for (Real value : alpha)
  {
    out += value * value;
  }
  return 0.5 * prior_weight * out;
}

void printFinalSummary(const TaoResult&      result,
                       const TimeTrajectory& tr,
                       Index                 physical_params,
                       Real                  forward_obs_objective,
                       Real                  prior_weight)
{
  const Real prior = coefficientPrior(result.prm, prior_weight);
  cout << "\nFinal summary\n";
  cout << "  coefficients: " << result.prm.size() << '\n';
  cout << "  physical parameters: " << physical_params << '\n';
  cout << "  TAO converged: " << (result.converged() ? "yes" : "no")
       << ", reason = " << result.reason
       << ", iterations = " << result.its << '\n';
  cout << "  final reduced objective: " << result.value
       << ", |grad| = " << sqrt(result.grad_norm_squared) << '\n';
  cout << "  final forward objective: " << forward_obs_objective + prior
       << ", obs = " << forward_obs_objective
       << ", prior = " << prior << '\n';
  cout << "  alpha:";
  for (Real value : result.prm)
  {
    cout << ' ' << value;
  }
  cout << '\n';
  cout << "  trajectory levels: " << tr.numLevels() << '\n';
}

bool taoFailed(const TaoResult& result)
{
  return !result.converged()
         && result.reason != TAO_DIVERGED_MAXITS;
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

  AppNsEnVar           app(prm);
  const inv::ElemRange elems = inv::mpiElemRange(app.space.mesh().numElems());
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

  EnsembleBasis basis =
      ensembleBasis(prm.inv.ens, app.prm0, app.problem.numParams());

  if (rank == 0)
  {
    prog.phase("optimization");
  }

  auto solveReduced =
      [&](auto& reduced) -> TaoResult
  {
    TaoOptimizer optimizer(reduced, PETSC_COMM_WORLD);
    configOptimizer(optimizer, prm.inv.opt);

    optimizer.setMonitor(
        [&prog, &app, &basis, rank](
            const TaoIterationInfo& info,
            const Vector<Real>&     current_alpha)
        {
          if (rank != 0)
          {
            return;
          }
          const Vector<Real> current_prm =
              physicalParams(basis, current_alpha);
          prog.optStep(info, current_alpha, current_prm, app.lyt);
        });

    TaoResult            out;
    const PetscErrorCode ierr =
        optimizer.solve(zeroCoefficients(basis), out);
    inv::checkPetsc(ierr, "TAO solve");
    return out;
  };

  EnsembleBasis obs_basis =
      observationEnsembleBasis(prm.inv.ens,
                               observationVectorSize(obj.data),
                               basis.numCoefficients());
  LinearObservationEnVarFunctional reduced(
      obs_basis,
      flattenObservations(obj.data),
      prm.inv.alpha,
      prm.inv.ens.prior_weight);
  TaoResult          result     = solveReduced(reduced);
  const Vector<Real> result_prm = physicalParams(basis, result.prm);

  if (rank == 0)
  {
    prog.phase("final forward solve");
  }

  TimeTrajectory tr;
  reduced_state_solver->solve(result_prm, tr);
  if (rank == 0)
  {
    prog.phase("write visualization");

    const Vector<Real> ctr_prm = controlParams(app.lyt, result_prm);
    writeResultViz(app.mesh,
                   app.space,
                   app.ctr,
                   tr,
                   ctr_prm,
                   app.ctr_time_stencils,
                   app.dt,
                   {prm.fwd.output.base},
                   0.0);

    const Real forward_obs_objective = obj.err.value(tr, result_prm);
    printFinalSummary(result,
                      tr,
                      result_prm.size(),
                      forward_obs_objective,
                      prm.inv.ens.prior_weight);
    cout << "  visualization: " << prm.fwd.output.base << ".xdmf\n";
  }

  return taoFailed(result) ? 2 : 0;
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
      if (opts.max_iterations)
      {
        prm.inv.opt.max_iterations = *opts.max_iterations;
      }
      if (opts.abs_tol)
      {
        prm.inv.opt.abs_tol = *opts.abs_tol;
      }
      if (opts.rel_tol)
      {
        prm.inv.opt.rel_tol = *opts.rel_tol;
      }
      if (opts.step_tol)
      {
        prm.inv.opt.step_tol = *opts.step_tol;
      }
      exit_code = run(prm);
    }
  }
  catch (const exception& e)
  {
    cerr << FEMX_NAVIER_EN_VAR_APP_NAME << " failed: " << e.what()
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
