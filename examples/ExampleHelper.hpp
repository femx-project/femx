#pragma once

#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/common/Types.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/runtime/LinearSystemFactory.hpp>
#include <femx/state/Residual.hpp>

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
 * @brief Small helpers for femx examples.
 */
class ExampleHelper
{
public:
  /**
   * @brief Bind solver, execution device, and output directory.
   *
   * @param[in] solver - Solver used by the example.
   * @param[in] device - Execution device used by the example.
   * @param[in] out_dir - Directory for generated output files.
   */
  ExampleHelper(runtime::SolverType      solver,
                runtime::ExecutionDevice device,
                std::string              out_dir)
    : solver_(solver),
      device_(device),
      out_dir_(std::move(out_dir))
  {
  }

  /** @brief Return the `solver/execution-device` display name. */
  std::string name() const
  {
    return std::string(runtime::name(solver_)) + "/"
           + runtime::name(device_);
  }

  /**
   * @brief Compute the Host residual norm.
   *
   * @param[in] op - Residual operator.
   * @param[in] state - State at which to evaluate the residual.
   * @param[in] prm - Parameter vector.
   * @param[in,out] ctx - Host execution context.
   * @return Euclidean norm of the residual.
   */
  Real resNorm(const state::HostResidual&          op,
               const HostVector<Real>&             state,
               const HostVector<Real>&             prm,
               linalg::Context<MemorySpace::Host>& ctx) const
  {
    HostVector<Real> residual;
    op.res(state, prm, residual, ctx);
    return std::sqrt(ctx.vectors().squaredNorm(residual.view()));
  }

#if defined(FEMX_HAS_CUDA)
  /**
   * @brief Compute the Device residual norm.
   *
   * The operation synchronizes `ctx` before returning the Host result.
   *
   * @param[in] op - Residual operator.
   * @param[in] state - Device state at which to evaluate the residual.
   * @param[in] prm - Device parameter vector.
   * @param[in,out] ctx - Device execution context.
   * @return Euclidean norm of the residual.
   */
  Real resNorm(
      const state::DeviceResidual&          op,
      const DeviceVector<Real>&             state,
      const DeviceVector<Real>&             prm,
      linalg::Context<MemorySpace::Device>& ctx) const
  {
    auto&              vec_handler = ctx.vectors();
    DeviceVector<Real> residual;
    op.res(state, prm, residual, ctx);

    DeviceVector<Real> norm2(1);
    vec_handler.squaredNorm(residual.view(), norm2.view());

    HostVector<Real> host_norm2;
    vec_handler.copy(norm2, host_norm2);
    ctx.sync();
    return std::sqrt(host_norm2[0]);
  }
#endif

  /**
   * @brief Build an output path containing solver and execution-device names.
   *
   * @param[in] stem - Problem-specific file stem.
   * @return Output path without a file extension.
   */
  std::string outputBase(const std::string& stem) const
  {
    const std::filesystem::path dir(out_dir_);
    const std::string           file =
        stem + "-" + runtime::name(solver_) + "-" + runtime::name(device_);
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
  runtime::SolverType      solver_;  ///< Solver used by the example.
  runtime::ExecutionDevice device_;  ///< Execution device used by the example.
  std::string              out_dir_; ///< Directory for generated output files.
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
