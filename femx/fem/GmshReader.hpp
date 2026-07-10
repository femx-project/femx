#pragma once

#include <string>

#include <femx/fem/Mesh.hpp>

namespace femx
{

/**
 * @brief Reader for Gmsh mesh files.
 */
class GmshReader
{
public:
  /** @brief Read a Gmsh `.msh` file into a Mesh object. */
  static Mesh read(const std::string& path);
};

} // namespace femx
