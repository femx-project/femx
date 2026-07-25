#include "CommandLine.hpp"

#include <ostream>
#include <stdexcept>

#include <femx/runtime/Cli.hpp>

namespace femx::apps::ns_forward
{

CommandLineOptions parseCommandLine(int   argc,
                                    char* argv[],
                                    bool  allow_unknown_options)
{
  CommandLineOptions options;

  for (int i = 1; i < argc; ++i)
  {
    const std::string key(argv[i]);
    if (key == "-h" || key == "--help")
    {
      options.help = true;
      return options;
    }
    if (key == "--config" || key == "-config")
    {
      options.config_file = runtime::requireValue(argc, argv, i, key);
      continue;
    }
    if (!allow_unknown_options)
    {
      throw std::runtime_error("Unknown option: " + key);
    }
  }

  if (options.config_file.empty())
  {
    throw std::runtime_error("Missing required option: --config FILE");
  }

  return options;
}

void printUsage(std::ostream&             out,
                const std::string&        executable,
                const std::string&        option_suffix,
                const Array<std::string>& extra_lines)
{
  out << "Usage: " << executable << " --config FILE" << option_suffix << '\n';
  for (const std::string& line : extra_lines)
  {
    out << line << '\n';
  }
}

} // namespace femx::apps::ns_forward
