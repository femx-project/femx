#pragma once

#include <string>

#include <femx/fem/Mesh.hpp>

namespace femx
{

/**
 * @brief Reader for Gmsh `.msh` files.
 *
 * GmshReader converts supported mesh entities and physical tags into a femx
 * Mesh.
 */
class GmshReader
{
public:
  /** @brief Read a mesh from disk and return a femx Mesh. */
  static Mesh read(const std::string& path);
};

} // namespace femx
