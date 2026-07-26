#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonForward.hpp"
#include <femx/linalg/Context.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/native/HostContext.hpp>
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/runtime/LinearSystemFactory.hpp>

#if defined(FEMX_RESOLVE_USE_CUDA)
#include "PoissonComponents.hpp"
#include <femx/assembly/CudaAssembly.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaJacobian.hpp>
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

void solveHost(const ExampleHelper&         helper,
               const PoissonForwardProblem& problem,
               HostVector<Real>&            x,
               Real&                        res_norm)
{
  HostCsrMatrix    mat(problem.map().pattern());
  HostVector<Real> rhs;
  problem.assemble(mat, rhs);

  ReSolveLinearSolver solver;
  linalg::HostContext ctx;
  solver.solve(mat, rhs, x, ctx);
  res_norm = helper.resNorm(mat, rhs, x, ctx);
}

#if defined(FEMX_RESOLVE_USE_CUDA)
void solveDevice(const ExampleHelper&         helper,
                 const PoissonForwardProblem& problem,
                 HostVector<Real>&            x,
                 Real&                        res_norm)
{
  auto system = runtime::makeDeviceLinearSystem(
      runtime::SolverType::ReSolve,
      std::make_unique<ReSolveLinearSolver>());
  auto& ctx         = dynamic_cast<linalg::CudaContext&>(system->context());
  auto& jac         = dynamic_cast<linalg::CudaJacobian&>(system->jacobian());
  auto& vec_handler = ctx.vectors();

  fem::DeviceGeometry              geom;
  fem::DeviceElementQuadratureData data;
  assembly::DeviceAssemblyMap      map;
  assembly::DeviceBoundaryMap      bc_map;
  copy(problem.geom(), geom, ctx);
  copy(problem.elementData(), data, ctx);
  assembly::copy(problem.map(), map, ctx);
  assembly::copy(problem.bcMap(), bc_map, ctx);

  DeviceVector<Real> state(problem.numDofs());
  DeviceVector<Real> res;
  DeviceVector<Real> rhs(problem.numDofs());
  DeviceVector<Real> bc_vals;
  vec_handler.copy(problem.bcVals(), bc_vals);

  jac.begin(problem.map().pattern());
  assembly::assemble(PoissonComponents<MemorySpace::Device>(data.view()),
                     geom,
                     map,
                     state,
                     res,
                     jac,
                     ctx);
  vec_handler.axpby(-1.0, res.view(), 0.0, rhs.view());
  jac.eliminateColumns(bc_map.view().constrained_rows, bc_vals.view(), rhs.view());

  DeviceVector<Real> sol;
  system->solve(rhs.view(), sol);

  res_norm = helper.resNorm(jac.matrix(), rhs, sol, ctx);
  vec_handler.copy(sol, x);
  ctx.sync();
}
#endif

int run(const Options& opts)
{
  constexpr auto        solver = runtime::SolverType::ReSolve;
  ExampleHelper         helper(solver, opts.execution_device, outputDir());
  PoissonForwardProblem problem(opts);

  HostVector<Real> x;
  Real             res_norm = 0.0;
  if (opts.execution_device == runtime::ExecutionDevice::Host)
  {
    solveHost(helper, problem, x, res_norm);
  }
  else
  {
#if defined(FEMX_RESOLVE_USE_CUDA)
    solveDevice(helper, problem, x, res_norm);
#else
    throw std::runtime_error(
        "Device Poisson execution requires a CUDA-enabled ReSolve build");
#endif
  }

  printReport(std::cout,
              helper.name(),
              problem,
              problem.errorReport(x),
              res_norm);

  if (opts.write_output)
  {
    const std::string base = helper.outputBase(outputStem(opts));
    problem.writeSolution(x, base);
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
