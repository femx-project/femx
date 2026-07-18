#pragma once

#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/common/Types.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/LinearOperator.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::examples
{

/** @brief Return true when the standard help flag is present. */
inline bool hasHelp(int argc, char* const argv[])
{
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h")
    {
      return true;
    }
  }
  return false;
}

/** @brief Return a lower-case name for a femx memory space. */
inline const char* memspaceName(MemorySpace memspace)
{
  switch (memspace)
  {
  case MemorySpace::Host:
    return "cpu";
  case MemorySpace::Device:
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
  /** @brief Bind display names and the output directory for one backend. */
  ExampleHelper(std::string solver,
                MemorySpace memspace,
                std::string out_dir)
    : solver_(std::move(solver)),
      memspace_(memspace),
      out_dir_(std::move(out_dir))
  {
  }

  /** @brief Return the `solver/memory-space` display name. */
  std::string name() const
  {
    return solver_ + "/" + memspaceName(memspace_);
  }

  /** @brief Compute `||A x - rhs||_2` through a linear-operator interface. */
  Real resNorm(const linalg::LinearOperator& A,
               const HostVector&             rhs,
               const HostVector&             x) const
  {
    if (rhs.size() != A.numRows() || x.size() != A.numCols())
    {
      throw std::runtime_error("Residual dimensions are inconsistent");
    }

    HostVector Ax;
    A.apply(x, Ax);

    Real norm2 = 0.0;
    for (Index i = 0; i < rhs.size(); ++i)
    {
      const Real r  = Ax[i] - rhs[i];
      norm2        += r * r;
    }
    return std::sqrt(norm2);
  }

  /** @brief Compute `||A x - rhs||_2` directly from a host CSR matrix. */
  Real resNorm(const HostCsrMatrix& A,
               const HostVector&    rhs,
               const HostVector&    x) const
  {
    if (rhs.size() != A.rows() || x.size() != A.cols())
    {
      throw std::runtime_error("Residual dimensions are inconsistent");
    }

    Real norm2 = 0.0;
    for (Index row = 0; row < A.rows(); ++row)
    {
      Real val = -rhs[row];
      for (Index k = A.rowPtrData()[row]; k < A.rowPtrData()[row + 1]; ++k)
      {
        val += A.valsData()[k] * x[A.colIndData()[k]];
      }
      norm2 += val * val;
    }
    return std::sqrt(norm2);
  }

  /** @brief Build an output path containing solver and memory-space names. */
  std::string outputBase(const std::string& stem) const
  {
    const std::filesystem::path dir(out_dir_);
    const std::string           file = stem + "-" + solver_ + "-" + memspaceName(memspace_);
    return (dir / file).string();
  }

  /** @brief Print the path of a visualization file after it has been written. */
  void printVisualizationPath(const std::string& base,
                              const std::string& extension = ".vtu") const
  {
    std::cout << "  wrote visualization: " << base << extension << '\n';
  }

private:
  std::string solver_;
  MemorySpace memspace_;
  std::string out_dir_;
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
