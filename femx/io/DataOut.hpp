#pragma once

#include <string>
#include <vector>

#include <femx/io/Hdf5Writer.hpp>

namespace femx
{

class Mesh;
class Vector;

class DataOut
{
public:
  void attachMesh(const Mesh& mesh);
  void addNodalField(const std::string& name, const Vector& values);
  void clearFields();

  void write(const std::string& basename) const;

private:
  const Mesh*                         mesh_{nullptr};
  std::vector<Hdf5Writer::NodalField> nodal_fields_;
};

} // namespace femx
