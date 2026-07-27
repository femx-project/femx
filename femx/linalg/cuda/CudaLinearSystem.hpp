#pragma once

#include <memory>

#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/LinearSystem.hpp>
#include <femx/linalg/cuda/CudaContext.hpp>
#include <femx/linalg/cuda/CudaSystemMatrix.hpp>

namespace femx::linalg
{

/**
 * @brief Own a CUDA context, Device system matrix, and Device-native solver.
 */
class CudaLinearSystem final : public LinearSystem<MemorySpace::Device>
{
public:
  /**
   * @brief Construct a CUDA system using a Device-native solver.
   *
   * @param[in] solver - Solver whose ownership is transferred to the system.
   */
  explicit CudaLinearSystem(
      std::unique_ptr<LinearSolver<MemorySpace::Device>> solver);

  ~CudaLinearSystem() override;

  CudaLinearSystem(const CudaLinearSystem&)            = delete;
  CudaLinearSystem& operator=(const CudaLinearSystem&) = delete;
  CudaLinearSystem(CudaLinearSystem&&)                 = delete;
  CudaLinearSystem& operator=(CudaLinearSystem&&)      = delete;

  Context<MemorySpace::Device>&      context() noexcept override;
  SystemMatrix<MemorySpace::Device>& matrix() noexcept override;
  void                               solve(ConstView rhs, Vector& solution) override;
  void                               solveT(ConstView rhs, Vector& solution) override;

private:
  CudaContext                                        ctx_;
  CudaSystemMatrix                                   mat_;
  std::unique_ptr<LinearSolver<MemorySpace::Device>> solver_;
  DeviceVector<Real>                                 rhs_;
};

} // namespace femx::linalg
