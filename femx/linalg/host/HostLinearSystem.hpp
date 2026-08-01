#pragma once

#include <memory>

#include <femx/linalg/LinearSystem.hpp>
#include <femx/linalg/host/HostContext.hpp>
#include <femx/linalg/host/HostSystemMatrix.hpp>

namespace femx::linalg
{

/**
 * @brief Own a serial Host context, CSR system matrix, and Host solver.
 */
class HostLinearSystem final : public LinearSystem<MemorySpace::Host>
{
public:
  /**
   * @brief Construct a Host system using a dense fallback solver.
   */
  HostLinearSystem();

  /**
   * @brief Construct a Host system using a native Host solver.
   *
   * @param[in] solver - Solver whose ownership is transferred to the system.
   */
  explicit HostLinearSystem(std::unique_ptr<Solver> solver);

  ~HostLinearSystem() override;

  HostLinearSystem(const HostLinearSystem&)            = delete;
  HostLinearSystem& operator=(const HostLinearSystem&) = delete;
  HostLinearSystem(HostLinearSystem&&)                 = delete;
  HostLinearSystem& operator=(HostLinearSystem&&)      = delete;

  /**
   * @copydoc LinearSystem::context()
   */
  Context<MemorySpace::Host>& context() noexcept override;

  /**
   * @copydoc LinearSystem::matrix()
   */
  SystemMatrix<MemorySpace::Host>& matrix() noexcept override;

  /**
   * @copydoc LinearSystem::solve()
   */
  void solve(VectorView rhs, Vector& x) override;

  /**
   * @copydoc LinearSystem::solveT()
   */
  void solveT(VectorView rhs, Vector& x) override;

private:
  HostContext             ctx_;
  HostSystemMatrix        mat_;
  std::unique_ptr<Solver> solver_;
  HostVector<Real>        rhs_;
};

} // namespace femx::linalg
