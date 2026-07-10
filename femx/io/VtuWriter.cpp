#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>

#include <femx/fem/Element.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/io/VtuWriter.hpp>

namespace femx
{
namespace
{

bool isLittleEndian()
{
  const uint16_t value = 1;
  return *reinterpret_cast<const unsigned char*>(&value) == 1;
}

std::string escapeXml(const std::string& text)
{
  std::string out;
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
  throw std::runtime_error("VtuWriter received unsupported elem shape");
}

std::string base64Encode(const unsigned char* data,
                         size_t               size)
{
  static constexpr char table[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  std::string out;
  out.reserve(((size + 2) / 3) * 4);
  for (size_t i = 0; i < size; i += 3)
  {
    const unsigned int b0    = data[i];
    const unsigned int b1    = (i + 1 < size) ? data[i + 1] : 0;
    const unsigned int b2    = (i + 2 < size) ? data[i + 2] : 0;
    const unsigned int value = (b0 << 16) | (b1 << 8) | b2;

    out.push_back(table[(value >> 18) & 0x3f]);
    out.push_back(table[(value >> 12) & 0x3f]);
    out.push_back(i + 1 < size ? table[(value >> 6) & 0x3f] : '=');
    out.push_back(i + 2 < size ? table[value & 0x3f] : '=');
  }
  return out;
}

uint64_t dataBytes(Index count, size_t value_size, const char* label)
{
  if (count < 0)
  {
    throw std::runtime_error(std::string("VtuWriter ") + label + " size is negative");
  }
  const auto size = static_cast<uint64_t>(count);
  if (size > std::numeric_limits<uint64_t>::max() / value_size)
  {
    throw std::runtime_error(std::string("VtuWriter ") + label + " data is too large");
  }
  return size * static_cast<uint64_t>(value_size);
}

Index bufferSize(uint64_t bytes, const char* label)
{
  if (bytes > static_cast<uint64_t>(std::numeric_limits<Index>::max())
                  - static_cast<uint64_t>(sizeof(bytes)))
  {
    throw std::runtime_error(std::string("VtuWriter ") + label + " data is too large");
  }
  return static_cast<Index>(sizeof(bytes) + bytes);
}

template <typename T>
std::string binaryBase64(const T* data, Index count, const char* label)
{
  const uint64_t        bytes = dataBytes(count, sizeof(T), label);
  Vector<unsigned char> buffer(bufferSize(bytes, label));
  std::memcpy(buffer.data(), &bytes, sizeof(bytes));
  if (bytes > 0)
  {
    std::memcpy(buffer.data() + sizeof(bytes),
                data,
                static_cast<size_t>(bytes));
  }
  return base64Encode(buffer.data(), static_cast<size_t>(buffer.size()));
}

template <typename T>
std::string binaryBase64(const Vector<T>& values, const char* label)
{
  return binaryBase64(values.data(), values.size(), label);
}

Vector<Real> pointValues(const Mesh& mesh)
{
  Vector<Real> values(mesh.numNodes() * 3);
  for (Index in = 0; in < mesh.numNodes(); ++in)
  {
    const auto& node   = mesh.node(in);
    values[3 * in]     = node[0];
    values[3 * in + 1] = node[1];
    values[3 * in + 2] = node[2];
  }
  return values;
}

Vector<Real> pointValues(const Vector<Point3>& points)
{
  Vector<Real> values(points.size() * 3);
  for (Index in = 0; in < points.size(); ++in)
  {
    values[3 * in]     = points[in][0];
    values[3 * in + 1] = points[in][1];
    values[3 * in + 2] = points[in][2];
  }
  return values;
}

Vector<int64_t> connectivityValues(const Mesh& mesh)
{
  Index num_node_ids = 0;
  for (Index ie = 0; ie < mesh.numElems(); ++ie)
  {
    num_node_ids += mesh.elem(ie).numNodes();
  }

  Vector<int64_t> values;
  values.reserve(num_node_ids);
  for (Index ie = 0; ie < mesh.numElems(); ++ie)
  {
    const auto& elem = mesh.elem(ie);
    for (Index i = 0; i < elem.numNodes(); ++i)
    {
      values.push_back(static_cast<int64_t>(elem.nodeIds()[i]));
    }
  }
  return values;
}

Vector<int64_t> vertexConnectivityValues(Index num_points)
{
  Vector<int64_t> values(num_points);
  for (Index i = 0; i < num_points; ++i)
  {
    values[i] = static_cast<int64_t>(i);
  }
  return values;
}

Vector<int64_t> offsetValues(const Mesh& mesh)
{
  Vector<int64_t> values(mesh.numElems());
  int64_t         offset = 0;
  for (Index ie = 0; ie < mesh.numElems(); ++ie)
  {
    offset     += static_cast<int64_t>(mesh.elem(ie).numNodes());
    values[ie]  = offset;
  }
  return values;
}

Vector<int64_t> vertexOffsetValues(Index num_points)
{
  Vector<int64_t> values(num_points);
  for (Index i = 0; i < num_points; ++i)
  {
    values[i] = static_cast<int64_t>(i + 1);
  }
  return values;
}

Vector<uint8_t> cellTypeValues(const Mesh& mesh)
{
  Vector<uint8_t> values(mesh.numElems());
  for (Index ie = 0; ie < mesh.numElems(); ++ie)
  {
    values[ie] = static_cast<uint8_t>(vtkCellType(mesh.elem(ie).shape()));
  }
  return values;
}

Vector<uint8_t> vertexCellTypeValues(Index num_points)
{
  return Vector<uint8_t>(num_points, 1);
}

void checkField(Index num_points, const VtuWriter::PointField& field)
{
  if (field.name.empty())
  {
    throw std::runtime_error("VtuWriter point field name must not be empty");
  }
  if (field.num_components <= 0)
  {
    throw std::runtime_error("VtuWriter point field component count must be positive");
  }
  if (field.vals == nullptr)
  {
    throw std::runtime_error("VtuWriter point field has null values");
  }
  if (field.vals->size() != num_points * field.num_components)
  {
    throw std::runtime_error("VtuWriter point field size does not match points");
  }
}

void writeUnstructuredGrid(const std::string&        fname,
                           const Vector<Real>&       points,
                           const Vector<int64_t>&    connectivity,
                           const Vector<int64_t>&    offsets,
                           const Vector<uint8_t>&    cell_types,
                           Index                     num_points,
                           Index                     num_cells,
                           const Vector<VtuWriter::PointField>& fields)
{
  std::ofstream out(fname, std::ios::binary);
  if (!out)
  {
    throw std::runtime_error("Failed to open VTU file: " + fname);
  }

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
         "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
  out << "  <UnstructuredGrid>\n";
  out << "    <Piece NumberOfPoints=\"" << num_points
      << "\" NumberOfCells=\"" << num_cells << "\">\n";

  out << "      <Points>\n";
  out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
         "format=\"binary\">"
      << binaryBase64(points, "points")
      << "</DataArray>\n";
  out << "      </Points>\n";

  out << "      <Cells>\n";
  out << "        <DataArray type=\"Int64\" Name=\"connectivity\" "
         "format=\"binary\">"
      << binaryBase64(connectivity, "connectivity")
      << "</DataArray>\n";

  out << "        <DataArray type=\"Int64\" Name=\"offsets\" "
         "format=\"binary\">"
      << binaryBase64(offsets, "offsets")
      << "</DataArray>\n";

  out << "        <DataArray type=\"UInt8\" Name=\"types\" "
         "format=\"binary\">"
      << binaryBase64(cell_types, "cell types")
      << "</DataArray>\n";
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
      out << " format=\"binary\">"
          << binaryBase64(*field.vals, field.name.c_str())
          << "</DataArray>\n";
    }
    out << "      </PointData>\n";
  }

  out << "    </Piece>\n";
  out << "  </UnstructuredGrid>\n";
  out << "</VTKFile>\n";
}

} // namespace

void VtuWriter::writePointData(const std::string&        fname,
                               const Mesh&               mesh,
                               const Vector<PointField>& fields) const
{
  if (!isLittleEndian())
  {
    throw std::runtime_error(
        "VtuWriter currently writes little-endian VTU files only");
  }
  if (mesh.numNodes() == 0 || mesh.numElems() == 0)
  {
    throw std::runtime_error("VtuWriter needs a non-empty mesh");
  }

  for (const auto& field : fields)
  {
    checkField(mesh.numNodes(), field);
  }

  const Vector<Real>    points       = pointValues(mesh);
  const Vector<int64_t> connectivity = connectivityValues(mesh);
  const Vector<int64_t> offsets      = offsetValues(mesh);
  const Vector<uint8_t> cell_types   = cellTypeValues(mesh);

  writeUnstructuredGrid(fname,
                        points,
                        connectivity,
                        offsets,
                        cell_types,
                        mesh.numNodes(),
                        mesh.numElems(),
                        fields);
}

void VtuWriter::writePointCloud(const std::string&        fname,
                                const Vector<Point3>&     points,
                                const Vector<PointField>& fields) const
{
  if (!isLittleEndian())
  {
    throw std::runtime_error(
        "VtuWriter currently writes little-endian VTU files only");
  }
  if (points.empty())
  {
    throw std::runtime_error("VtuWriter point cloud needs at least one point");
  }

  for (const auto& field : fields)
  {
    checkField(points.size(), field);
  }

  const Vector<Real>    point_data   = pointValues(points);
  const Vector<int64_t> connectivity = vertexConnectivityValues(points.size());
  const Vector<int64_t> offsets      = vertexOffsetValues(points.size());
  const Vector<uint8_t> cell_types   = vertexCellTypeValues(points.size());

  writeUnstructuredGrid(fname,
                        point_data,
                        connectivity,
                        offsets,
                        cell_types,
                        points.size(),
                        points.size(),
                        fields);
}

} // namespace femx
