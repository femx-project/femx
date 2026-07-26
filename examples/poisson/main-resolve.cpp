#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "../ExampleHelper.hpp"
#include "PoissonProblem.hpp"
#include "PoissonResidual.hpp"
#include <femx/linalg/native/HostLinearSystem.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/state/StateSolver.hpp>

#if defined(FEMX_RESOLVE_USE_CUDA)
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaLinearSystem.hpp>
#endif

using namespace femx;
using namespace femx::examples;
using namespace femx::examples::poisson;
using namespace femx::linalg;

#ifndef FEMX_POISSON_APP_NAME
#define FEMX_POISSON_APP_NAME "poisson-resolve"
#endif

namespace
{

Real solveHost(const ExampleHelper&  helper,
               const PoissonProblem& prob,
               HostVector<Real>&     x)
{
  auto             solver = std::make_unique<ReSolveLinearSolver>();
  HostLinearSystem system(std::move(solver));

  HostPoissonResidual          res(prob);
  state::HostLinearStateSolver state_solver(res, system);

  state_solver.solve(x);

  const Real rnorm = helper.resNorm(res, x, system.context());

  return rnorm;
}

#if defined(FEMX_RESOLVE_USE_CUDA)
Real solveDevice(const ExampleHelper&  helper,
                 const PoissonProblem& prob,
                 HostVector<Real>&     x)
{
  auto             solver = std::make_unique<ReSolveLinearSolver>();
  CudaLinearSystem system(std::move(solver));

  auto& ctx = static_cast<linalg::CudaContext&>(system.context());

  DevicePoissonResidual                         res(prob, ctx);
  state::LinearStateSolver<MemorySpace::Device> state_solver(res, system);

  DeviceVector<Real> d_x;
  state_solver.solve(d_x);

  const Real rnorm = helper.resNorm(res, d_x, ctx);
  ctx.vectors().copy(d_x, x);
  ctx.sync();

  return rnorm;
}
#endif

int run(const Options& opts)
{
  constexpr auto solver_type = runtime::SolverType::ReSolve;
  ExampleHelper  helper(solver_type, opts.memspace, outputDir());
  PoissonProblem prob(opts);

  HostVector<Real> x;
  Real             rnorm;
  if (opts.memspace == MemorySpace::Host)
  {
    rnorm = solveHost(helper, prob, x);
  }
  else
  {
#if defined(FEMX_RESOLVE_USE_CUDA)
    rnorm = solveDevice(helper, prob, x);
#else
    throw std::runtime_error(
        "Device Poisson execution requires a CUDA-enabled ReSolve build");
#endif
  }

  printReport(std::cout,
              helper.name(),
              prob,
              prob.errorReport(x),
              rnorm);

  if (opts.write_output)
  {
    const std::string base = helper.outputBase(outputStem(opts));
    prob.writeSolution(x, base);
    helper.printVisualizationPath(base);
  }

  return 0;
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    if (examples::hasHelp(argc, argv))
    {
      printUsage(FEMX_POISSON_APP_NAME, false);
      return 0;
    }
    return run(parseOptions(argc, argv, false));
  }
  catch (const std::exception& e)
  {
    return examples::reportError(FEMX_POISSON_APP_NAME, e);
  }
}
