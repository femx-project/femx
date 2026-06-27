#include "BuildInfo.hpp"

#include <fstream>

#include "Output.hpp"

namespace femx::runtime
{

void writeBuildInfo(const std::filesystem::path& directory,
                    const BuildInfo&             info,
                    const std::string&           file_name)
{
  std::ofstream out = openOutputFile(directory, file_name);
  for (const auto& [key, value] : info.entries)
  {
    out << key << ": " << value << '\n';
  }
}

} // namespace femx::runtime
