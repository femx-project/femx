#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::examples::poisson
{

class PoissonForwardProblem;

struct CudaSolveResult
{
  HostVector sol;
  Real       res_norm{0.0};
};

CudaSolveResult solveCuda(const PoissonForwardProblem& problem);

} // namespace femx::examples::poisson
