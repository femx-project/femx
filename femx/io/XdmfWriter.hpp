#pragma once

#include <string>

#include <femx/linalg/Vector.hpp>

namespace femx
{

class Mesh;

class XdmfWriter
{
public:
  void write(const std::string&         fname,
             const std::string&         hdf5_filename,
             const Mesh&                mesh,
             const Vector<std::string>& nodal_field_names) const;
};

} // namespace femx
