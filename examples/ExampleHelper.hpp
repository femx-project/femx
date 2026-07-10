#pragma once

#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/common/Types.hpp>
#include <femx/common/Workspace.hpp>
#include <femx/linalg/LinearOperator.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::examples
{

/** @brief Return a lower-case name for a femx workspace backend. */
inline const char* workspaceName(WorkspaceType backend)
{
  switch (backend)
  {
  case WorkspaceType::Cpu:
    return "cpu";
  case WorkspaceType::Cuda:
    return "cuda";
  }
  return "unknown";
}

/**
 * @brief Small helpers for femx examples.
 */
class ExampleHelper
{
public:
  ExampleHelper(std::string   solver_name,
                WorkspaceType backend,
                std::string   output_dir)
    : solver_name_(std::move(solver_name)),
      backend_(backend),
      output_dir_(std::move(output_dir))
  {
  }

  WorkspaceType backend() const noexcept
  {
    return backend_;
  }

  std::string hardwareBackend() const
  {
    return workspaceName(backend_);
  }

  std::string backendName() const
  {
    return solver_name_ + "/" + hardwareBackend();
  }

  const std::string& outputDirectory() const noexcept
  {
    return output_dir_;
  }

  Real residualNorm(const linalg::LinearOperator& A,
                    const Vector<Real>&           rhs,
                    const Vector<Real>&           x) const
  {
    if (rhs.size() != A.numRows() || x.size() != A.numCols())
    {
      throw std::runtime_error("Residual dimensions are inconsistent");
    }

    Vector<Real> Ax;
    A.apply(x, Ax);

    Real norm2 = 0.0;
    for (Index i = 0; i < rhs.size(); ++i)
    {
      const Real r  = Ax[i] - rhs[i];
      norm2        += r * r;
    }
    return std::sqrt(norm2);
  }

  std::string outputBase(const std::string& output_stem) const
  {
    const std::filesystem::path output_dir(output_dir_);
    const std::string           filename = output_stem + "-" + solver_name_ + "-"
                                 + hardwareBackend();
    return (output_dir / filename).string();
  }

  /** @brief Print the path of a visualization file after it has been written. */
  void printVisualizationPath(const std::string& output_base,
                              const std::string& extension = ".vtu") const
  {
    std::cout << "  wrote visualization: " << output_base << extension << '\n';
  }

private:
  std::string   solver_name_;
  WorkspaceType backend_;
  std::string   output_dir_;
};

/** @brief Print a standard example error message and return failure status. */
inline int reportError(const char*           app_name,
                       const std::exception& e,
                       std::ostream&         err = std::cerr)
{
  err << app_name << ": " << e.what() << '\n';
  return 1;
}

} // namespace femx::examples
