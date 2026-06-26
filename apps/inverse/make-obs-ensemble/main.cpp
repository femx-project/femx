#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <Helper.hpp>
#include <PetscCommon.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/petsc/KspLinearSolver.hpp>
#include <femx/linalg/petsc/PETScMatrixOperator.hpp>
#include <femx/linalg/petsc/PETScVectorBuilder.hpp>
#include <femx/state/EnsembleBasis.hpp>
#include <femx/state/TimeLinearStateSolver.hpp>
#include <femx/state/TimeTrajectory.hpp>

using namespace std;
using namespace femx;
using namespace femx::linalg;
using namespace femx::navier_en_var;
using namespace femx::state;
namespace inv = femx::inverse;

#ifndef FEMX_MAKE_OBS_ENSEMBLE_APP_NAME
#define FEMX_MAKE_OBS_ENSEMBLE_APP_NAME "make-obs-ensemble"
#endif

namespace
{

struct AppOptions
{
  string          config_file;
  string          mean_output_file;
  string          perturbations_output_file;
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
    if (key == "--obs-mean-output")
    {
      opts.mean_output_file = inv::requireValue(argc, argv, i, key);
      continue;
    }
    if (key == "--obs-perturbations-output")
    {
      opts.perturbations_output_file =
          inv::requireValue(argc, argv, i, key);
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
    throw runtime_error("Unknown option: " + key);
  }
  return opts;
}

void printUsage(ostream& out)
{
  out << "Usage: " << FEMX_MAKE_OBS_ENSEMBLE_APP_NAME
      << " --config FILE [--obs-mean-output FILE]"
      << " [--obs-perturbations-output FILE] [--steps N]"
      << " [PETSc options]\n";
  out << "  Output files default to inverse.ensemble.obs_*_file when set.\n";
}

string outputMeanFile(const AppOptions& opts,
                      const Params&     prm)
{
  if (!opts.mean_output_file.empty())
  {
    return opts.mean_output_file;
  }
  if (!prm.inv.ens.obs_mean_file.empty())
  {
    return prm.inv.ens.obs_mean_file;
  }
  return "obs_mean.txt";
}

string outputPerturbationsFile(const AppOptions& opts,
                               const Params&     prm)
{
  if (!opts.perturbations_output_file.empty())
  {
    return opts.perturbations_output_file;
  }
  if (!prm.inv.ens.obs_perturbations_file.empty())
  {
    return prm.inv.ens.obs_perturbations_file;
  }
  return "obs_perturbations.txt";
}

Vector<Real> modePhysicalParams(const EnsembleBasis& basis,
                                Index                col)
{
  if (col < 0 || col >= basis.numCoefficients())
  {
    throw runtime_error("ensemble mode index is out of range");
  }

  Vector<Real> alpha(basis.numCoefficients());
  alpha.setZero();
  alpha[col] = 1.0;

  Vector<Real> out;
  basis.apply(alpha, out);
  return out;
}

Vector<Real> observationVector(problem::TimeObservationOperator&   obs,
                               const problem::TimeObservationData& layout,
                               TimeStateSolver&                    solver,
                               const Vector<Real>&                 prm,
                               Real                                dt)
{
  TimeTrajectory tr;
  solver.solve(prm, tr);
  return flattenObservations(
      sampleObservationData(obs, layout, tr, prm, dt));
}

void printStep(PetscMPIInt   rank,
               const string& label,
               Index         step,
               Index         total)
{
  if (rank != 0)
  {
    return;
  }
  const Index stride = max<Index>(Index{1}, total / 5);
  if (step % stride != 0 && step < total)
  {
    return;
  }

  cout << "\r    " << label << " step " << setw(4) << step
       << " / " << setw(4) << total << flush;
  if (step >= total)
  {
    cout << '\n';
  }
}

void printSummary(const AppNsEnVar&    app,
                  const EnsembleBasis& basis,
                  Index                nobs,
                  const string&        mean_file,
                  const string&        perturbations_file)
{
  cout << "make-obs-ensemble\n";
  cout << "  physical parameters: " << app.lyt.ntot << '\n';
  cout << "  coefficients: " << basis.numCoefficients() << '\n';
  cout << "  observation vector size: " << nobs << '\n';
  cout << "  observation levels: " << app.obs_data.numLevels() << '\n';
  cout << "  observations per level: " << app.obs_data.numObservations()
       << '\n';
  cout << "  obs mean: " << mean_file << '\n';
  cout << "  obs perturbations: " << perturbations_file << '\n';
}

int run(Params&           prm,
        const AppOptions& opts)
{
  PetscMPIInt rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  checkInverseRunParams(prm);

  AppNsEnVar           app(prm);
  const inv::ElemRange elems =
      inv::mpiElemRange(app.space.mesh().numElems());
  app.fem.setElemRange(elems.begin, elems.end);

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
      cout << "  initial guess\n";
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

  string label;
  state_solver.setStepMonitor(
      [&label, rank](Index step, Index total)
      {
        printStep(rank, label, step, total);
      });

  unique_ptr<InitialVelocityStateSolver> init_state_solver;
  TimeStateSolver*                       forward = &state_solver;
  if (app.lyt.hasInitialVelocity())
  {
    init_state_solver =
        make_unique<InitialVelocityStateSolver>(
            state_solver,
            app.init_vdofs,
            app.lyt,
            std::move(app.x0));
    forward = init_state_solver.get();
  }

  Objective     obj(prm, app);
  EnsembleBasis basis = ensembleBasis(prm.inv.ens, app.prm0, app.problem.numParams());

  if (rank == 0)
  {
    cout << "  mean forward\n";
  }
  label                    = "mean";
  const Vector<Real> hmean = observationVector(obj.op, obj.data, *forward, basis.mean(), app.dt);

  DenseMatrix y(hmean.size(), basis.numCoefficients());
  for (Index col = 0; col < basis.numCoefficients(); ++col)
  {
    if (rank == 0)
    {
      cout << "  perturbation forward " << col + 1 << " / "
           << basis.numCoefficients() << '\n';
    }
    label                       = "mode " + to_string(col + 1);
    const Vector<Real> prm_mode = modePhysicalParams(basis, col);
    const Vector<Real> hmode    = observationVector(obj.op, obj.data, *forward, prm_mode, app.dt);
    if (hmode.size() != hmean.size())
    {
      throw runtime_error("observation perturbation size mismatch");
    }
    for (Index i = 0; i < hmean.size(); ++i)
    {
      y(i, col) = hmode[i] - hmean[i];
    }
  }

  state_solver.clearStepMonitor();

  if (rank == 0)
  {
    const string mean_file          = outputMeanFile(opts, prm);
    const string perturbations_file = outputPerturbationsFile(opts, prm);
    inv::writeVector(mean_file, hmean, inv::VectorFileHeader::size);
    inv::writeMatrix(perturbations_file, y);
    printSummary(app,
                 basis,
                 hmean.size(),
                 mean_file,
                 perturbations_file);
  }

  return 0;
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
      exit_code = run(prm, opts);
    }
  }
  catch (const exception& e)
  {
    cerr << FEMX_MAKE_OBS_ENSEMBLE_APP_NAME << " failed: "
         << e.what() << '\n';
    exit_code = 1;
  }

  ierr = PetscFinalize();
  if (ierr != PETSC_SUCCESS && exit_code == 0)
  {
    return 1;
  }
  return exit_code;
}
