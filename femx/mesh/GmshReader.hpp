#pragma once

#include <string>

#include <femx/mesh/Mesh.hpp>

namespace femx
{

class GmshReader
{
public:
  static Mesh read(const std::string& path);
};

} // namespace femx
