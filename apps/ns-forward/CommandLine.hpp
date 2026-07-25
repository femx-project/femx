#pragma once

#include <iosfwd>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::apps::ns_forward
{

struct CommandLineOptions
{
  std::string config_file;  ///< Input JSON configuration file.
  bool        help = false; ///< Print help and exit.
};

/**
 * @brief Parse ns-forward command-line options.
 *
 * @param[in] argc - Argument count.
 * @param[in] argv - Argument values.
 * @param[in] allow_unknown_options - Preserve options owned by a backend.
 * @return Parsed command-line options.
 * @throws std::runtime_error - If required or supported options are invalid.
 */
CommandLineOptions parseCommandLine(int   argc,
                                    char* argv[],
                                    bool  allow_unknown_options);

/**
 * @brief Print ns-forward command-line usage.
 *
 * @param[out] out - Output stream.
 * @param[in] executable - Executable name.
 * @param[in] option_suffix - Additional backend option syntax.
 * @param[in] extra_lines - Additional usage lines.
 */
void printUsage(std::ostream&             out,
                const std::string&        executable,
                const std::string&        option_suffix = {},
                const Array<std::string>& extra_lines   = {});

} // namespace femx::apps::ns_forward
