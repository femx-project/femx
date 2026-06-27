#include "Output.hpp"

#include <stdexcept>

namespace femx::runtime
{

void ensureDirectory(const std::filesystem::path& dir)
{
  if (!dir.empty())
  {
    std::filesystem::create_directories(dir);
  }
}

void ensureParentDirectory(const std::filesystem::path& path)
{
  ensureDirectory(path.parent_path());
}

std::ofstream openOutputFile(const std::filesystem::path& path)
{
  ensureParentDirectory(path);

  std::ofstream out(path);
  if (!out)
  {
    throw std::runtime_error("Failed to open output file: "
                             + path.string());
  }
  return out;
}

std::ofstream openOutputFile(const std::filesystem::path& directory,
                             const std::string&           file_name)
{
  ensureDirectory(directory);
  return openOutputFile(directory / file_name);
}

} // namespace femx::runtime
