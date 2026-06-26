#include <fstream>
#include <stdexcept>
#include <string>

#include <femx/fem/Mesh.hpp>
#include <femx/io/XdmfWriter.hpp>

using namespace std;

namespace femx
{
namespace
{

Index nodesPerElem(const Mesh& mesh)
{
  if (mesh.numElems() == 0)
  {
    throw runtime_error("XdmfWriter needs a non-empty mesh");
  }
  return mesh.elems().front().numNodes();
}

string topologyType(const Mesh& mesh)
{
  const Index nodes = nodesPerElem(mesh);
  if (nodes == 3)
  {
    return "Triangle";
  }
  if (nodes == 4)
  {
    return "Quadrilateral";
  }
  throw runtime_error("XdmfWriter supports triangle and quadrilateral elems for now");
}

string filenameOnly(const string& path)
{
  const size_t pos = path.find_last_of("/\\");
  if (pos == string::npos)
  {
    return path;
  }
  return path.substr(pos + 1);
}

} // namespace

void XdmfWriter::write(const string&         fname,
                       const string&         hdf5_filename,
                       const Mesh&           mesh,
                       const Vector<string>& nodal_field_names) const
{
  ofstream out(fname);
  if (!out)
  {
    throw runtime_error("Failed to open XDMF file for writing: " + fname);
  }

  const string h5_ref = filenameOnly(hdf5_filename);

  out << R"(<?xml version="1.0" ?>)" << '\n';
  out << R"(<Xdmf Version="3.0">)" << '\n';
  out << "  <Domain>\n";
  out << R"(    <Grid Name="mesh" GridType="Uniform">)" << '\n';
  const Index cn = nodesPerElem(mesh);
  out << R"(      <Topology TopologyType=")" << topologyType(mesh)
      << R"(" NumberOfElements=")"
      << mesh.numElems() << "\">\n";
  out << R"(        <DataItem Dimensions=")" << mesh.numElems()
      << " " << cn << R"(" NumberType="Int" Precision="4" Format="HDF">)"
      << h5_ref << ":/Mesh/Topology</DataItem>\n";
  out << "      </Topology>\n";
  out << R"(      <Geometry GeometryType="XYZ">)" << '\n';
  out << R"(        <DataItem Dimensions=")" << mesh.numNodes()
      << R"( 3" NumberType="Float" Precision="8" Format="HDF">)"
      << h5_ref << ":/Mesh/Geometry</DataItem>\n";
  out << "      </Geometry>\n";

  for (const string& name : nodal_field_names)
  {
    out << R"(      <Attribute Name=")" << name
        << R"(" AttributeType="Scalar" Center="Node">)" << '\n';
    out << R"(        <DataItem Dimensions=")" << mesh.numNodes()
        << R"(" NumberType="Float" Precision="8" Format="HDF">)"
        << h5_ref << ":/Data/" << name << "</DataItem>\n";
    out << "      </Attribute>\n";
  }

  out << "    </Grid>\n";
  out << "  </Domain>\n";
  out << "</Xdmf>\n";
}

} // namespace femx
