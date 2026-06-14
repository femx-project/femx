#pragma once

#include <string>
#include <vector>

namespace femx
{

class Mesh;

class XdmfWriter
{
public:
  void write(const std::string&              filename,
             const std::string&              hdf5_filename,
             const Mesh&                     mesh,
             const std::vector<std::string>& nodal_field_names) const;
};

} // namespace femx
