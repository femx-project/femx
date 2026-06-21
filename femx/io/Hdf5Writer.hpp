#pragma once

#include <string>
#include <vector>

#include <femx/common/Types.hpp>

namespace femx
{

class Mesh;
template <typename T>
class Vector;

class Hdf5Writer
{
public:
  struct NodalField
  {
    std::string         name;
    const Vector<Real>* values{nullptr};
  };

  void write(const std::string&             filename,
             const Mesh&                    mesh,
             const std::vector<NodalField>& nodal_fields) const;
};

} // namespace femx
