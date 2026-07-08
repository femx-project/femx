#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>

#include <femx/fem/Element.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/io/VtuWriter.hpp>

using namespace std;

namespace femx
{
namespace
{

string escapeXml(const string& text)
{
  string out;
  out.reserve(text.size());
  for (char ch : text)
  {
    switch (ch)
    {
    case '&':
      out += "&amp;";
      break;
    case '<':
      out += "&lt;";
      break;
    case '>':
      out += "&gt;";
      break;
    case '"':
      out += "&quot;";
      break;
    case '\'':
      out += "&apos;";
      break;
    default:
      out += ch;
      break;
    }
  }
  return out;
}

int vtkCellType(Element::Shape shape)
{
  switch (shape)
  {
  case Element::Shape::Segment:
    return 3;
  case Element::Shape::Triangle:
    return 5;
  case Element::Shape::Quadrilateral:
    return 9;
  case Element::Shape::Tetrahedron:
    return 10;
  case Element::Shape::Hexahedron:
    return 12;
  case Element::Shape::Unknown:
    break;
  }
  throw runtime_error("VtuWriter received unsupported elem shape");
}

void checkField(const Mesh& mesh, const VtuWriter::PointField& field)
{
  if (field.name.empty())
  {
    throw runtime_error("VtuWriter point field name must not be empty");
  }
  if (field.num_components <= 0)
  {
    throw runtime_error("VtuWriter point field component count must be positive");
  }
  if (field.vals == nullptr)
  {
    throw runtime_error("VtuWriter point field has null values");
  }
  if (field.vals->size() != mesh.numNodes() * field.num_components)
  {
    throw runtime_error("VtuWriter point field size does not match mesh nodes");
  }
}

} // namespace

void VtuWriter::writePointData(const string&             fname,
                               const Mesh&               mesh,
                               const Vector<PointField>& fields) const
{
  if (mesh.numNodes() == 0 || mesh.numElems() == 0)
  {
    throw runtime_error("VtuWriter needs a non-empty mesh");
  }

  for (const auto& field : fields)
  {
    checkField(mesh, field);
  }

  ofstream out(fname);
  if (!out)
  {
    throw runtime_error("Failed to open VTU file: " + fname);
  }

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
         "byte_order=\"LittleEndian\">\n";
  out << "  <UnstructuredGrid>\n";
  out << "    <Piece NumberOfPoints=\"" << mesh.numNodes()
      << "\" NumberOfCells=\"" << mesh.numElems() << "\">\n";

  out << "      <Points>\n";
  out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
         "format=\"ascii\">\n";
  out << setprecision(numeric_limits<Real>::max_digits10);
  for (Index in = 0; in < mesh.numNodes(); ++in)
  {
    const auto& node = mesh.node(in);
    out << "          " << node[0] << ' ' << node[1] << ' ' << node[2]
        << '\n';
  }
  out << "        </DataArray>\n";
  out << "      </Points>\n";

  out << "      <Cells>\n";
  out << "        <DataArray type=\"Int64\" Name=\"connectivity\" "
         "format=\"ascii\">\n";
  for (Index ie = 0; ie < mesh.numElems(); ++ie)
  {
    const auto& elem = mesh.elem(ie);
    out << "          ";
    for (Index i = 0; i < elem.numNodes(); ++i)
    {
      if (i > 0)
      {
        out << ' ';
      }
      out << elem.nodeIds()[i];
    }
    out << '\n';
  }
  out << "        </DataArray>\n";

  out << "        <DataArray type=\"Int64\" Name=\"offsets\" "
         "format=\"ascii\">\n";
  Index offset = 0;
  for (Index ie = 0; ie < mesh.numElems(); ++ie)
  {
    offset += mesh.elem(ie).numNodes();
    out << "          " << offset << '\n';
  }
  out << "        </DataArray>\n";

  out << "        <DataArray type=\"UInt8\" Name=\"types\" "
         "format=\"ascii\">\n";
  for (Index ie = 0; ie < mesh.numElems(); ++ie)
  {
    out << "          " << vtkCellType(mesh.elem(ie).shape()) << '\n';
  }
  out << "        </DataArray>\n";
  out << "      </Cells>\n";

  if (!fields.empty())
  {
    out << "      <PointData>\n";
    for (const auto& field : fields)
    {
      out << "        <DataArray type=\"Float64\" Name=\""
          << escapeXml(field.name) << "\"";
      if (field.num_components > 1)
      {
        out << " NumberOfComponents=\"" << field.num_components << "\"";
      }
      out << " format=\"ascii\">\n";
      for (Index in = 0; in < mesh.numNodes(); ++in)
      {
        out << "          ";
        for (Index comp = 0; comp < field.num_components; ++comp)
        {
          if (comp > 0)
          {
            out << ' ';
          }
          out << (*field.vals)[in * field.num_components + comp];
        }
        out << '\n';
      }
      out << "        </DataArray>\n";
    }
    out << "      </PointData>\n";
  }

  out << "    </Piece>\n";
  out << "  </UnstructuredGrid>\n";
  out << "</VTKFile>\n";
}

} // namespace femx
