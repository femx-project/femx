#include "Cli.hpp"

#include <stdexcept>

namespace femx::runtime
{

std::string requireValue(int                argc,
                         char**             argv,
                         int&               index,
                         const std::string& key)
{
  if (index + 1 >= argc)
  {
    throw std::runtime_error("Missing value for " + key);
  }
  return std::string(argv[++index]);
}

} // namespace femx::runtime
