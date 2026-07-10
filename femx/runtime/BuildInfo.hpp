#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace femx::runtime
{

struct BuildInfo
{
  std::vector<std::pair<std::string, std::string>> entries; ///< Key-value entries.
};

void writeBuildInfo(const std::filesystem::path& directory,
                    const BuildInfo&             info,
                    const std::string&           file_name = "build-info.txt");

} // namespace femx::runtime
