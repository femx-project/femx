#pragma once

#include <string>
#include <vector>

namespace femx
{

class Mesh;
class Vector;

class Hdf5Writer
{
public:
  struct NodalField
  {
    std::string   name;
    const Vector* values{nullptr};
  };

  void write(const std::string&             filename,
             const Mesh&                    mesh,
             const std::vector<NodalField>& nodal_fields) const;
};

} // namespace femx
