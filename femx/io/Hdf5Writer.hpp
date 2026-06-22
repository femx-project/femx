#pragma once

#include <string>

#include <femx/common/Types.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{

class Mesh;

class Hdf5Writer
{
public:
  struct NodalField
  {
    std::string         name;
    const Vector<Real>* vals{nullptr};
  };

  void write(const std::string&        fname,
             const Mesh&               mesh,
             const Vector<NodalField>& nodal_fields) const;
};

} // namespace femx
