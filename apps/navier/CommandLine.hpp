#pragma once

#include <iosfwd>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/common/Vector.hpp>

namespace femx::apps::navier
{

/** @brief Store parsed command-line options for the Navier-Stokes app. */
struct CommandLineOptions
{
  std::string config_file;  ///< Input JSON configuration file.
  bool        help = false; ///< Print help and exit.
};

/**
 * @brief Parse navier command-line options.
 *
 * @param[in] argc - Argument count.
 * @param[in] argv - Argument values.
 * @param[in] allow_unknown_options - Preserve options owned by a solver.
 * @return Parsed command-line options.
 * @throws std::runtime_error - If required or supported options are invalid.
 */
CommandLineOptions parseCommandLine(int   argc,
                                    char* argv[],
                                    bool  allow_unknown_options);

/**
 * @brief Print navier command-line usage.
 *
 * @param[out] out - Output stream.
 * @param[in] executable - Executable name.
 * @param[in] option_suffix - Additional solver option syntax.
 * @param[in] extra_lines - Additional usage lines.
 */
void printUsage(std::ostream&                  out,
                const std::string&             executable,
                const std::string&             option_suffix = {},
                const HostVector<std::string>& extra_lines   = {});

} // namespace femx::apps::navier
