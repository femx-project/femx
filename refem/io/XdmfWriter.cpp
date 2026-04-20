#include <refem/io/XdmfWriter.hpp>

#include <fstream>
#include <stdexcept>
#include <string>

#include <refem/mesh/Mesh.hpp>

namespace refem
{
namespace
{

std::string filenameOnly(const std::string& path)
{
  const std::size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos)
  {
    return path;
  }
  return path.substr(pos + 1);
}

} // namespace

void XdmfWriter::write(const std::string&              filename,
                       const std::string&              hdf5_filename,
                       const Mesh&                     mesh,
                       const std::vector<std::string>& nodal_field_names) const
{
  std::ofstream out(filename);
  if (!out)
  {
    throw std::runtime_error("Failed to open XDMF file for writing: " +
                             filename);
  }

  const std::string h5_ref = filenameOnly(hdf5_filename);

  out << R"(<?xml version="1.0" ?>)" << '\n';
  out << R"(<Xdmf Version="3.0">)" << '\n';
  out << "  <Domain>\n";
  out << R"(    <Grid Name="mesh" GridType="Uniform">)" << '\n';
  out << R"(      <Topology TopologyType="Quadrilateral" NumberOfElements=")"
      << mesh.numElems() << "\">\n";
  out << R"(        <DataItem Dimensions=")" << mesh.numElems()
      << R"( 4" NumberType="Int" Precision="4" Format="HDF">)"
      << h5_ref << ":/Mesh/Topology</DataItem>\n";
  out << "      </Topology>\n";
  out << R"(      <Geometry GeometryType="XYZ">)" << '\n';
  out << R"(        <DataItem Dimensions=")" << mesh.numNodes()
      << R"( 3" NumberType="Float" Precision="8" Format="HDF">)"
      << h5_ref << ":/Mesh/Geometry</DataItem>\n";
  out << "      </Geometry>\n";

  for (const std::string& name : nodal_field_names)
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

} // namespace refem
