#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "../ExampleHelper.hpp"
#include "PoissonForward.hpp"
#include <femx/linalg/resolve/ReSolveLinearSolver.hpp>
#include <femx/runtime/LinearSystemFactory.hpp>
#include <femx/state/StateSolver.hpp>

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
  auto system = runtime::makeHostLinearSystem(
      runtime::SolverType::ReSolve,
      std::make_unique<ReSolveLinearSolver>());
  state::HostLinearStateSolver solver(problem, *system);
  const HostVector<Real>       prm;
  solver.solve(prm, x);
  res_norm = helper.resNorm(problem, x, prm, system->context());
}

#if defined(FEMX_RESOLVE_USE_CUDA)
class DevicePoissonResidual final : public state::DeviceResidual
{
public:
  using Vec = DeviceVector<Real>;
  using Jac = linalg::Jacobian<MemorySpace::Device>;
  using Ctx = linalg::Context<MemorySpace::Device>;

  DevicePoissonResidual(const PoissonForwardProblem& problem,
                        linalg::CudaContext&         ctx)
    : dims_{problem.numDofs(), 0, problem.numDofs()},
      pattern_(problem.map().pattern())
  {
    fem::copy(problem.geom(), geom_, ctx);
    fem::copy(problem.elementData(), data_, ctx);
    assembly::copy(problem.map(), map_, ctx);
    assembly::copy(problem.bcMap(), bc_map_, ctx);
    ctx.vectors().copy(problem.bcVals(), bc_vals_);
    ctx.sync();
  }

  state::Dimensions dims() const override
  {
    return dims_;
  }

  const HostCsrPattern& hostPattern() const override
  {
    return pattern_;
  }

  void res(const Vec& state,
           const Vec& prm,
           Vec&       out,
           Ctx&       base_ctx) const override
  {
    checkVectors(state, prm);
    auto& ctx = dynamic_cast<linalg::CudaContext&>(base_ctx);
    assembly::assembleResidual(
        PoissonComponents<MemorySpace::Device>(data_.view()),
        geom_,
        map_,
        state,
        out,
        ctx);
    assembly::replaceRes(
        bc_map_, state.view(), bc_vals_.view(), out.view(), ctx);
  }

  void assembleStateJac(const Vec& state,
                        const Vec& prm,
                        Jac&       out,
                        Ctx&       base_ctx) const override
  {
    checkVectors(state, prm);
    auto& ctx = dynamic_cast<linalg::CudaContext&>(base_ctx);
    auto& jac = dynamic_cast<linalg::CudaJacobian&>(out);
    Vec   unused;
    assembly::assemble(
        PoissonComponents<MemorySpace::Device>(data_.view()),
        geom_,
        map_,
        state,
        unused,
        jac,
        ctx);
    jac.replaceRows(bc_map_.view().constrained_rows, 1.0);
  }

  void applyParamJacT(const Vec& state,
                      const Vec& prm,
                      const Vec& adj,
                      Vec&       out,
                      Ctx&) const override
  {
    checkVectors(state, prm);
    if (adj.size() != dims_.num_res)
    {
      throw std::runtime_error(
          "Poisson adjoint vector has incompatible size");
    }
    out.resize(0);
  }

private:
  void checkVectors(const Vec& state, const Vec& prm) const
  {
    if (state.size() != dims_.num_states || !prm.empty())
    {
      throw std::runtime_error(
          "Poisson residual vectors have incompatible sizes");
    }
  }

  state::Dimensions                dims_;
  HostCsrPattern                   pattern_;
  fem::DeviceGeometry              geom_;
  fem::DeviceElementQuadratureData data_;
  assembly::DeviceAssemblyMap      map_;
  assembly::DeviceBoundaryMap      bc_map_;
  DeviceVector<Real>               bc_vals_;
};

void solveDevice(const ExampleHelper&         helper,
                 const PoissonForwardProblem& problem,
                 HostVector<Real>&            x,
                 Real&                        res_norm)
{
  auto system = runtime::makeDeviceLinearSystem(
      runtime::SolverType::ReSolve,
      std::make_unique<ReSolveLinearSolver>());

  auto&                 ctx = dynamic_cast<linalg::CudaContext&>(system->context());
  DevicePoissonResidual residual(problem, ctx);

  state::LinearStateSolver<MemorySpace::Device> solver(residual, *system);

  const DeviceVector<Real> prm;
  DeviceVector<Real>       sol;
  solver.solve(prm, sol);

  res_norm = helper.resNorm(residual, sol, prm, system->context());
  ctx.vectors().copy(sol, x);
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
