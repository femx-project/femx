#pragma once

#include <memory>

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
  explicit CudaLinearSystem(std::unique_ptr<Solver> solver);

  ~CudaLinearSystem() override;

  CudaLinearSystem(const CudaLinearSystem&)            = delete;
  CudaLinearSystem& operator=(const CudaLinearSystem&) = delete;
  CudaLinearSystem(CudaLinearSystem&&)                 = delete;
  CudaLinearSystem& operator=(CudaLinearSystem&&)      = delete;

  /**
   * @brief Return the system-owned CUDA execution context.
   */
  Context<MemorySpace::Device>& context() noexcept override;

  /**
   * @brief Return the system-owned CUDA matrix.
   */
  SystemMatrix<MemorySpace::Device>& matrix() noexcept override;

  /**
   * @brief Solve the assembled CUDA system.
   *
   * @param[in]  rhs - Right-hand side view.
   * @param[out] x - Solution vector.
   */
  void solve(ConstView rhs, Vector& x) override;

  /**
   * @brief Solve the transposed assembled CUDA system.
   *
   * @param[in]  rhs - Right-hand side view.
   * @param[out] x - Solution vector.
   */
  void solveT(ConstView rhs, Vector& x) override;

private:
  CudaContext             ctx_;    ///< Owned CUDA execution context.
  CudaSystemMatrix        mat_;    ///< Owned CUDA system matrix.
  std::unique_ptr<Solver> solver_; ///< Owned solver.
  DeviceVector<Real>      rhs_;    ///< Device copy of the right-hand side.
};

} // namespace femx::linalg
