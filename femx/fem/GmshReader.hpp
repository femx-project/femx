#pragma once

#include <string>

#include <femx/fem/Mesh.hpp>

namespace femx
{

class GmshReader
{
public:
  static Mesh read(const std::string& path);
};

} // namespace femx
