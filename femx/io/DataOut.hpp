#pragma once

#include <string>
#include <vector>

#include <femx/common/Types.hpp>
#include <femx/io/Hdf5Writer.hpp>

namespace femx
{

class Mesh;
template <typename T>
class Vector;

class DataOut
{
public:
  void attachMesh(const Mesh& mesh);
  void addNodalField(const std::string& name, const Vector<Real>& values);
  void clearFields();

  void write(const std::string& basename) const;

private:
  const Mesh*                         mesh_{nullptr};
  std::vector<Hdf5Writer::NodalField> nodal_fields_;
};

} // namespace femx
