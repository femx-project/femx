#pragma once

#include <filesystem>
#include <fstream>
#include <string>

namespace femx::runtime
{

void ensureDirectory(const std::filesystem::path& dir);

void ensureParentDirectory(const std::filesystem::path& path);

std::ofstream openOutputFile(const std::filesystem::path& path);

std::ofstream openOutputFile(const std::filesystem::path& directory,
                             const std::string&           file_name);

} // namespace femx::runtime
