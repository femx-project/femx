#pragma once

#include <string>

#include <refem/mesh/Mesh.hpp>

namespace refem
{

class GmshReader
{
public:
  static Mesh read(const std::string& path);
};

} // namespace refem
