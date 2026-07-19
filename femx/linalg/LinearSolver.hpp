#pragma once

#include <femx/linalg/Backend.hpp>

namespace femx::linalg
{

/** @brief Linear solve contract over one concrete execution backend. */
template <class Backend>
class LinearSolver
{
  static_assert(is_backend_v<Backend>,
                "LinearSolver requires a valid backend type");

public:
  using Matrix  = typename Backend::Mat;
  using Vector  = typename Backend::Vec;
  using Context = typename Backend::Ctx;

  virtual ~LinearSolver() = default;

  /** @brief Solve `mat * sol = rhs`. */
  virtual void solve(const Matrix& mat,
                     const Vector& rhs,
                     Vector&       sol,
                     Context&      ctx) = 0;

  /** @brief Solve `mat^T * sol = rhs`. */
  virtual void solveT(const Matrix& mat,
                      const Vector& rhs,
                      Vector&       sol,
                      Context&      ctx) = 0;
};

using HostCsrLinearSolver = LinearSolver<HostCsrBackend>;
using DeviceLinearSolver  = LinearSolver<CudaCsrBackend>;

} // namespace femx::linalg
