#pragma once

#include <string>

#include <femx/common/Types.hpp>
#include <femx/io/Hdf5Writer.hpp>

namespace femx
{

class Mesh;

class DataOut
{
public:
  void attachMesh(const Mesh& mesh);
  void addNodalField(const std::string& name, const Vector<Real>& vals);
  void clearFields();

  void write(const std::string& base) const;

private:
  const Mesh*                    mesh_{nullptr};
  Vector<Hdf5Writer::NodalField> nodal_fields_;
};

} // namespace femx
