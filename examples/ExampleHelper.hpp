#pragma once

#include <charconv>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/common/Types.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/runtime/LinearSystemSelection.hpp>
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
 * @brief Parse an example memory-space value.
 *
 * @param[in] val - Value supplied to `--memory-space`.
 * @return Parsed memory space.
 * @throws std::runtime_error - If `val` is neither `host` nor `device`.
 */
inline MemorySpace parseMemorySpace(const std::string& val)
{
  if (val == "host")
  {
    return MemorySpace::Host;
  }
  if (val == "device")
  {
    return MemorySpace::Device;
  }
  throw std::runtime_error(
      "--memory-space expects 'host' or 'device'");
}

/**
 * @brief Parse a `yes` or `no` example option.
 *
 * @param[in] val - Option value.
 * @param[in] option - Option name used in diagnostics.
 * @return `true` for `yes` and `false` for `no`.
 * @throws std::runtime_error - If `val` is neither `yes` nor `no`.
 */
inline bool parseYesNo(const std::string& val,
                       const std::string& option)
{
  if (val == "yes")
  {
    return true;
  }
  if (val == "no")
  {
    return false;
  }
  throw std::runtime_error(option + " expects 'yes' or 'no'");
}

/**
 * @brief Parse a positive example index.
 *
 * @param[in] val - Integer text to parse.
 * @param[in] option - Option name used in diagnostics.
 * @return Parsed positive index.
 * @throws std::runtime_error - If `val` is not a positive `Index`.
 */
inline Index parsePositiveIndex(const std::string& val,
                                const std::string& option)
{
  long long  parsed = 0;
  const auto res =
      std::from_chars(val.data(), val.data() + val.size(), parsed);
  if (res.ec != std::errc()
      || res.ptr != val.data() + val.size()
      || parsed <= 0
      || parsed > std::numeric_limits<Index>::max())
  {
    throw std::runtime_error(option + " must be a positive integer");
  }
  return static_cast<Index>(parsed);
}

/**
 * @brief Small helpers for femx examples.
 */
class ExampleHelper
{
public:
  /**
   * @brief Bind solver, memory space, and output directory.
   *
   * @param[in] solver - Solver used by the example.
   * @param[in] space - Memory space used by the example.
   * @param[in] out_dir - Directory for generated output files.
   */
  ExampleHelper(runtime::SolverType solver,
                MemorySpace         space,
                std::string         out_dir)
    : solver_(solver),
      space_(space),
      out_dir_(std::move(out_dir))
  {
  }

  /** @brief Return the `solver/memory-space` display name. */
  std::string name() const
  {
    return std::string(runtime::name(solver_)) + "/"
           + runtime::name(space_);
  }

  /**
   * @brief Compute the Host residual norm.
   *
   * @param[in] op - Residual operator.
   * @param[in] state - State at which to evaluate the residual.
   * @param[in,out] ctx - Host execution context.
   * @return Euclidean norm of the residual.
   */
  Real resNorm(const state::HostResidual&          op,
               const HostVector<Real>&             state,
               linalg::Context<MemorySpace::Host>& ctx) const
  {
    HostVector<Real>       res;
    const HostVector<Real> prm;
    op.assembleResidual(state, prm, res, ctx);
    return std::sqrt(ctx.vectors().squaredNorm(res.view()));
  }

#if defined(FEMX_HAS_CUDA)
  /**
   * @brief Compute the Device residual norm.
   *
   * The operation synchronizes `ctx` before returning the Host result.
   *
   * @param[in] op - Residual operator.
   * @param[in] state - Device state at which to evaluate the residual.
   * @param[in,out] ctx - Device execution context.
   * @return Euclidean norm of the residual.
   */
  Real resNorm(
      const state::DeviceResidual&          op,
      const DeviceVector<Real>&             state,
      linalg::Context<MemorySpace::Device>& ctx) const
  {
    auto&                    vec_handler = ctx.vectors();
    const DeviceVector<Real> prm;
    DeviceVector<Real>       res;
    op.assembleResidual(state, prm, res, ctx);

    DeviceVector<Real> norm2(1);
    vec_handler.squaredNorm(res.view(), norm2.view());

    HostVector<Real> h_norm2;
    vec_handler.copy(norm2, h_norm2);
    ctx.sync();
    return std::sqrt(h_norm2[0]);
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
    const std::string           file =
        stem + "-" + runtime::name(solver_) + "-" + runtime::name(space_);
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
  runtime::SolverType solver_;  ///< Solver used by the example.
  MemorySpace         space_;   ///< Memory space used by the example.
  std::string         out_dir_; ///< Directory for generated output files.
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
