#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <femx/io/VtiWriter.hpp>
#include <femx/linalg/Vector.hpp>

using namespace std;

namespace femx
{
namespace
{

bool isLittleEndian()
{
  const uint16_t value = 1;
  return *reinterpret_cast<const unsigned char*>(&value) == 1;
}

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

array<Index, 3> wholeExtentMax(const VtiWriter::Image& image)
{
  return {image.elem_counts[0] > 1 ? image.elem_counts[0] : 0,
          image.elem_counts[1] > 1 ? image.elem_counts[1] : 0,
          image.elem_counts[2] > 1 ? image.elem_counts[2] : 0};
}

Index numElems(const VtiWriter::Image& image)
{
  Index count = 1;
  for (Index elems : image.elem_counts)
  {
    if (elems <= 0)
    {
      throw runtime_error("VtiWriter elem counts must be positive");
    }
    if (count > numeric_limits<Index>::max() / elems)
    {
      throw runtime_error("VtiWriter elem count overflow");
    }
    count *= elems;
  }
  return count;
}

void checkFinite(const VtiWriter::Image& image)
{
  for (Index d = 0; d < 3; ++d)
  {
    if (!isfinite(image.origin[d]) || !isfinite(image.spacing[d]))
    {
      throw runtime_error("VtiWriter origin and spacing must be finite");
    }
  }
  if (image.time && !isfinite(*image.time))
  {
    throw runtime_error("VtiWriter time must be finite");
  }
}

uint64_t fieldBytes(const VtiWriter::ElemField& field)
{
  if (field.vals == nullptr)
  {
    throw runtime_error("VtiWriter elem field has null values");
  }
  const auto size = static_cast<uint64_t>(field.vals->size());
  if (size > numeric_limits<uint64_t>::max() / sizeof(Real))
  {
    throw runtime_error("VtiWriter elem field is too large");
  }
  return size * static_cast<uint64_t>(sizeof(Real));
}

string base64Encode(const unsigned char* data,
                    size_t               size)
{
  static constexpr char table[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  string out;
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

string fieldBinaryBase64(const VtiWriter::ElemField& field)
{
  const uint64_t        bytes = fieldBytes(field);
  Vector<unsigned char> buffer(static_cast<Index>(sizeof(bytes) + bytes));
  memcpy(buffer.data(), &bytes, sizeof(bytes));
  memcpy(buffer.data() + sizeof(bytes),
         field.vals->data(),
         static_cast<size_t>(bytes));
  return base64Encode(buffer.data(), buffer.size());
}

void checkFields(const Vector<VtiWriter::ElemField>& fields,
                 Index                               elems)
{
  if (fields.empty())
  {
    throw runtime_error("VtiWriter needs at least one elem field");
  }

  for (const auto& field : fields)
  {
    if (field.name.empty())
    {
      throw runtime_error("VtiWriter elem field name must not be empty");
    }
    if (field.num_components <= 0)
    {
      throw runtime_error(
          "VtiWriter elem field component count must be positive");
    }
    if (field.vals == nullptr)
    {
      throw runtime_error("VtiWriter elem field has null values");
    }
    if (field.vals->size() != elems * field.num_components)
    {
      throw runtime_error(
          "VtiWriter elem field size does not match image elem count");
    }
  }
}

string extentString(const array<Index, 3>& max_extent)
{
  ostringstream out;
  out << "0 " << max_extent[0] << " 0 " << max_extent[1] << " 0 "
      << max_extent[2];
  return out.str();
}

string point3String(const array<Real, 3>& vals)
{
  ostringstream out;
  out << setprecision(numeric_limits<Real>::max_digits10)
      << vals[0] << ' ' << vals[1] << ' ' << vals[2];
  return out.str();
}

} // namespace

void VtiWriter::writeElemData(const string&            fname,
                              const Image&             image,
                              const Vector<ElemField>& fields) const
{
  if (!isLittleEndian())
  {
    throw runtime_error(
        "VtiWriter currently writes little-endian VTI files only");
  }

  checkFinite(image);
  const Index elems = numElems(image);
  checkFields(fields, elems);

  ofstream out(fname, ios::binary);
  if (!out)
  {
    throw runtime_error("Failed to open VTI file: " + fname);
  }

  const auto max_extent = wholeExtentMax(image);
  const auto extent     = extentString(max_extent);

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\""
      << "LittleEndian\" header_type=\"UInt64\">\n";
  out << "  <ImageData WholeExtent=\"" << extent << "\" Origin=\""
      << point3String(image.origin) << "\" Spacing=\""
      << point3String(image.spacing) << "\">\n";
  if (image.time)
  {
    out << "    <FieldData>\n";
    out << "      <DataArray type=\"Float64\" Name=\"TimeValue\" "
           "NumberOfTuples=\"1\" format=\"ascii\">"
        << setprecision(numeric_limits<Real>::max_digits10)
        << *image.time << "</DataArray>\n";
    out << "    </FieldData>\n";
  }
  out << "    <Piece Extent=\"" << extent << "\">\n";
  out << "      <CellData>\n";
  for (size_t i = 0; i < fields.size(); ++i)
  {
    const auto& field = fields[i];
    out << "        <DataArray type=\"Float64\" Name=\""
        << escapeXml(field.name) << "\"";
    if (field.num_components > 1)
    {
      out << " NumberOfComponents=\"" << field.num_components << "\"";
    }
    out << " format=\"binary\">"
        << fieldBinaryBase64(field)
        << "</DataArray>\n";
  }
  out << "      </CellData>\n";
  out << "    </Piece>\n";
  out << "  </ImageData>\n";
  out << "</VTKFile>\n";
}

} // namespace femx
