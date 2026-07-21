#pragma once

#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/common/Context.hpp>
#include <femx/common/Types.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/linalg/handler/MatrixHandler.hpp>
#include <femx/linalg/handler/VectorHandler.hpp>

namespace femx::examples
{

/**
 * @brief Report whether the standard help flag is present.
 *
 * @param[in] argc - Number of command-line arguments.
 * @param[in] argv - Command-line argument values.
 * @return `true` when `--help` or `-h` is present.
 */
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

/**
 * @brief Return a lower-case name for a femx memory space.
 *
 * @param[in] memspace - Memory space to name.
 * @return `"cpu"` for Host, `"cuda"` for Device, or `"unknown"` otherwise.
 */
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
  /**
   * @brief Bind display names and the output directory for one backend.
   *
   * @param[in] solver - Solver name used in reports and output paths.
   * @param[in] memspace - Memory space used by the example.
   * @param[in] out_dir - Directory for generated output files.
   */
  ExampleHelper(std::string solver,
                MemorySpace memspace,
                std::string out_dir)
    : solver_(std::move(solver)),
      memspace_(memspace),
      out_dir_(std::move(out_dir))
  {
  }

  /**
   * @brief Return the `solver/memory-space` display name.
   *
   * @return Combined solver and memory-space name.
   */
  std::string name() const
  {
    return solver_ + "/" + memspaceName(memspace_);
  }

  /**
   * @brief Compute `||A x - rhs||_2` with Host linear algebra operations.
   *
   * @param[in] A - Host CSR matrix.
   * @param[in] rhs - Host right-hand side.
   * @param[in] x - Host solution vector.
   * @param[in,out] ctx - CPU execution context.
   * @return Euclidean norm of the algebraic residual.
   * @throws std::runtime_error - If the matrix and vector dimensions are
   * incompatible.
   */
  Real resNorm(const HostCsrMatrix& A,
               const HostVector&    rhs,
               const HostVector&    x,
               CpuContext&          ctx) const
  {
    if (rhs.size() != A.rows() || x.size() != A.cols())
    {
      throw std::runtime_error("Residual dimensions are inconsistent");
    }

    linalg::HostMatrixHandler mat_handler(ctx);
    linalg::HostVectorHandler vec_handler(ctx);

    HostVector residual;
    mat_handler.matvec(A, x.view(), residual);
    vec_handler.axpby(-1.0, rhs.view(), 1.0, residual.view());
    return std::sqrt(vec_handler.squaredNorm(residual.view()));
  }

#if defined(FEMX_HAS_CUDA)
  /**
   * @brief Compute `||A x - rhs||_2` with Device linear algebra operations.
   *
   * The operation synchronizes `ctx` before returning the Host result.
   *
   * @param[in] A - Device CSR matrix.
   * @param[in] rhs - Device right-hand side.
   * @param[in] x - Device solution vector.
   * @param[in,out] ctx - CUDA execution context.
   * @return Euclidean norm of the algebraic residual.
   * @throws std::runtime_error - If dimensions are incompatible or a CUDA
   * linear algebra operation fails.
   */
  Real resNorm(const DeviceCsrMatrix& A,
               const DeviceVector&    rhs,
               const DeviceVector&    x,
               CudaContext&           ctx) const
  {
    if (rhs.size() != A.rows() || x.size() != A.cols())
    {
      throw std::runtime_error("Residual dimensions are inconsistent");
    }

    linalg::CudaMatrixHandler mat_handler(ctx);
    linalg::CudaVectorHandler vec_handler(ctx);

    DeviceVector residual;
    mat_handler.matvec(A, x.view(), residual);
    vec_handler.axpby(-1.0, rhs.view(), 1.0, residual.view());

    DeviceVector norm2(1);
    vec_handler.squaredNorm(residual.view(), norm2.view());

    HostVector host_norm2;
    vec_handler.copy(norm2, host_norm2);
    ctx.sync();
    return std::sqrt(host_norm2[0]);
  }
#endif

  /**
   * @brief Build an output path containing solver and memory-space names.
   *
   * @param[in] stem - Problem-specific file stem.
   * @return Output path without a file extension.
   */
  std::string outputBase(const std::string& stem) const
  {
    const std::filesystem::path dir(out_dir_);
    const std::string           file = stem + "-" + solver_ + "-" + memspaceName(memspace_);
    return (dir / file).string();
  }

  /**
   * @brief Print the path of a visualization file after it has been written.
   *
   * @param[in] base - Output path without an extension.
   * @param[in] extension - Visualization file extension, including its dot.
   */
  void printVisualizationPath(const std::string& base,
                              const std::string& extension = ".vtu") const
  {
    std::cout << "  wrote visualization: " << base << extension << '\n';
  }

private:
  std::string solver_;   ///< Solver name used in reports and output paths.
  MemorySpace memspace_; ///< Memory space used by the example.
  std::string out_dir_;  ///< Directory for generated output files.
};

/**
 * @brief Print a standard example error message and return failure status.
 *
 * @param[in] app_name - Application name used as the message prefix.
 * @param[in] e - Exception whose message is reported.
 * @param[in,out] err - Stream that receives the error message.
 * @return Failure status code `1`.
 */
inline int reportError(const char*           app_name,
                       const std::exception& e,
                       std::ostream&         err = std::cerr)
{
  err << app_name << ": " << e.what() << '\n';
  return 1;
}

} // namespace femx::examples
