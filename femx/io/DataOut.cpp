#include <stdexcept>
#include <string>

#include <femx/io/DataOut.hpp>
#include <femx/io/XdmfWriter.hpp>

using namespace std;

namespace femx
{
namespace
{

string stripKnownExtension(string path)
{
  const auto strip = [&path](const string& ext)
  {
    if (path.size() >= ext.size() && path.compare(path.size() - ext.size(), ext.size(), ext) == 0)
    {
      path.resize(path.size() - ext.size());
      return true;
    }
    return false;
  };

  strip(".xdmf") || strip(".h5");
  return path;
}

} // namespace

void DataOut::attachMesh(const Mesh& mesh)
{
  mesh_ = &mesh;
}

void DataOut::addNodalField(const string& name, const Vector<Real>& vals)
{
  nodal_fields_.push_back({name, &vals});
}

void DataOut::clearFields()
{
  nodal_fields_.clear();
}

void DataOut::write(const string& base) const
{
  if (mesh_ == nullptr)
  {
    throw runtime_error("DataOut needs an attached mesh before writing");
  }

  const string root          = stripKnownExtension(base);
  const string hdf5_filename = root + ".h5";
  const string xdmf_filename = root + ".xdmf";

  Hdf5Writer hdf5_writer;
  hdf5_writer.write(hdf5_filename, *mesh_, nodal_fields_);

  Vector<string> field_names;
  field_names.reserve(nodal_fields_.size());
  for (const auto& field : nodal_fields_)
  {
    field_names.push_back(field.name);
  }

  XdmfWriter xdmf_writer;
  xdmf_writer.write(xdmf_filename, hdf5_filename, *mesh_, field_names);
}

} // namespace femx
