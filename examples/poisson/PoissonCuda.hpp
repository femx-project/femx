#pragma once

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::examples::poisson
{

class PoissonForwardProblem;

/** @brief Host-visible result returned by the CUDA Poisson solve. */
struct CudaSolveResult
{
  HostVector sol;           ///< Solution copied back from device memory.
  Real       res_norm{0.0}; ///< Euclidean residual norm.
};

/** @brief Assemble and solve the Poisson problem entirely on the CUDA path. */
CudaSolveResult solveCuda(const PoissonForwardProblem& problem);

} // namespace femx::examples::poisson
