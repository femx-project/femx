#include <cmath>

#include "PoissonCuda.hpp"
#include "PoissonForward.hpp"
#include "PoissonOperator.hpp"
#include <femx/assembly/CudaAssembly.hpp>
#include <femx/linalg/resolve/ReSolveDeviceSolver.hpp>

namespace femx::examples::poisson
{
namespace
{

Real resNorm(const HostCsrMatrix& mat,
             const HostVector&    rhs,
             const HostVector&    sol)
{
  Real norm2 = 0.0;
  for (Index row = 0; row < mat.rows(); ++row)
  {
    Real val = -rhs[row];
    for (Index k = mat.rowPtrData()[row]; k < mat.rowPtrData()[row + 1]; ++k)
    {
      val += mat.valsData()[k] * sol[mat.colIndData()[k]];
    }
    norm2 += val * val;
  }
  return std::sqrt(norm2);
}

} // namespace

CudaSolveResult solveCuda(const PoissonForwardProblem& problem)
{
  CudaContext ctx;

  fem::DeviceGeometry         geom;
  assembly::DeviceAssemblyMap map;
  assembly::DeviceBoundaryMap bc_map;
  copy(problem.geom(), geom, ctx);
  assembly::copy(problem.map(), map, ctx);
  assembly::copy(problem.bcMap(), bc_map, ctx);

  HostVector   zero_state(problem.numDofs(), 0.0);
  DeviceVector state;
  DeviceVector res;
  DeviceVector rhs(problem.numDofs());
  DeviceVector bc_vals;
  copy(zero_state, state, ctx);
  copy(problem.bcVals(), bc_vals, ctx);

  DeviceCsrMatrix mat(map.graph());
  assembly::assemble(PoissonQuadQ1Operator{},
                     geom,
                     map,
                     state,
                     res,
                     mat,
                     ctx);
  assembly::prepareForwardSolve(bc_map, mat, rhs, bc_vals, ctx);

  linalg::ReSolveOptions opts;
  opts.rtol    = 1.0e-11;
  opts.max_its = 500;
  opts.restart = 100;
  linalg::ReSolveDeviceSolver solver(opts);
  solver.setOperator(mat);

  DeviceVector sol;
  solver.solve(rhs, sol, ctx);

  CudaSolveResult out;
  HostCsrMatrix   hmat(problem.map().graph());
  HostVector      hrhs;
  copy(sol, out.sol, ctx);
  copy(mat, hmat, ctx);
  copy(rhs, hrhs, ctx);
  ctx.synchronize();
  out.res_norm = resNorm(hmat, hrhs, out.sol);
  return out;
}

} // namespace femx::examples::poisson
