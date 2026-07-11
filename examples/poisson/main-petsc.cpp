#include <petscksp.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonForward.hpp"
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/native/CsrAssemblyMatrix.hpp>
#include <femx/linalg/petsc/KspLinearSolver.hpp>
#include <femx/linalg/petsc/PETScAssemblyMatrix.hpp>
#include <femx/linalg/petsc/PETScVector.hpp>
#include <femx/runtime/PETScRuntime.hpp>

using namespace femx;
using namespace femx::examples::poisson;
using namespace femx::linalg;
using namespace femx::runtime;
using namespace femx::examples;

#ifndef FEMX_POISSON_APP_NAME
#define FEMX_POISSON_APP_NAME "poisson-petsc"
#endif

namespace
{

void copyToPETSc(const CsrMatrix& src, PETScAssemblyMatrix& dst)
{
  const Index* rp   = src.rowPtrData();
  const Index* ci   = src.colIndData();
  const Real*  vals = src.valuesData();

  for (Index row = 0; row < src.rows(); ++row)
  {
    for (Index k = rp[row]; k < rp[row + 1]; ++k)
    {
      if (vals[k] != 0.0)
      {
        dst.set(row, ci[k], vals[k]);
      }
    }
  }
}

int run(const Options& opts)
{
  if (opts.backend != WorkspaceType::Cpu)
  {
    throw std::runtime_error("PETSc Poisson backend supports only 'cpu'");
  }

  ExampleHelper         helper("petsc", opts.backend, outputDir());
  PoissonForwardProblem problem(opts);

  CsrAssemblyMatrix A(problem.pattern());
  Vector<Real>      rhs;
  problem.assemble(A, rhs);

  PETScVector layout(PETSC_COMM_WORLD);
  layout.resize(problem.numDofs());

  PETScAssemblyMatrix A_petsc(PETSC_COMM_WORLD);
  A_petsc.resize(problem.pattern(), layout);
  if (isRoot())
  {
    copyToPETSc(A.mat(), A_petsc);
  }
  A_petsc.finalize();

  KspLinearSolver solver(PETSC_COMM_WORLD);

  Vector<Real> x;
  solver.solve(A_petsc, rhs, x);

  if (isRoot())
  {
    printReport(std::cout,
                helper.backendName(),
                problem,
                problem.errorReport(x),
                helper.residualNorm(A_petsc, rhs, x));

    if (opts.write_output)
    {
      const std::string base = helper.outputBase(outputStem(opts));
      problem.writeSolution(x, base);
      helper.printVisualizationPath(base);
    }
  }

  return 0;
}

} // namespace

int main(int argc, char* argv[])
{
  int status = 0;
  try
  {
    PetscSession petsc(argc, argv);
    setSerialOpenMp();

    try
    {
      const bool help = [&]()
      {
        for (int i = 1; i < argc; ++i)
        {
          if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h")
          {
            return true;
          }
        }
        return false;
      }();

      if (help)
      {
        if (isRoot())
        {
          printUsage(FEMX_POISSON_APP_NAME, true);
        }
      }
      else
      {
        status = run(parseOptions(argc, argv, true));
      }
    }
    catch (const std::exception& e)
    {
      if (isRoot())
      {
        examples::reportError(FEMX_POISSON_APP_NAME, e);
      }
      status = 1;
    }

    const PetscErrorCode ierr = petsc.finalize();
    if (ierr != PETSC_SUCCESS && status == 0)
    {
      return 1;
    }
  }
  catch (const std::exception& e)
  {
    return examples::reportError(FEMX_POISSON_APP_NAME, e);
  }
  return status;
}
