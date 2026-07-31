#pragma once

#include <memory>

#include <femx/linalg/LinearSolver.hpp>
#include <femx/linalg/LinearSystem.hpp>
#include <femx/linalg/native/HostContext.hpp>
#include <femx/linalg/native/HostSystemMatrix.hpp>

namespace femx::linalg
{

/**
 * @brief Own a serial Host context, CSR system matrix, and Host solver.
 */
class HostLinearSystem final : public LinearSystem<MemorySpace::Host>
{
public:
  /** @brief Construct a Host system using a dense fallback solver. */
  HostLinearSystem();

  /**
   * @brief Construct a Host system using a native Host solver.
   *
   * @param[in] solver - Solver whose ownership is transferred to the system.
   */
  explicit HostLinearSystem(
      std::unique_ptr<LinearSolver<MemorySpace::Host>> solver);

  ~HostLinearSystem() override;

  HostLinearSystem(const HostLinearSystem&)            = delete;
  HostLinearSystem& operator=(const HostLinearSystem&) = delete;
  HostLinearSystem(HostLinearSystem&&)                 = delete;
  HostLinearSystem& operator=(HostLinearSystem&&)      = delete;

  Context<MemorySpace::Host>&      context() noexcept override;
  SystemMatrix<MemorySpace::Host>& matrix() noexcept override;
  void                             solve(ConstView rhs, Vector& x) override;
  void                             solveT(ConstView rhs, Vector& x) override;

private:
  HostContext                                      ctx_;
  HostSystemMatrix                                 mat_;
  std::unique_ptr<LinearSolver<MemorySpace::Host>> solver_;
  HostVector<Real>                                 rhs_;
};

} // namespace femx::linalg
