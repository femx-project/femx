#include <femx/io/VtiWriter.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace
{

bool isLittleEndian()
{
  const std::uint16_t value = 1;
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

std::array<Index, 3> wholeExtentMax(const VtiWriter::Image& image)
{
  return {image.cell_counts[0] > 1 ? image.cell_counts[0] : 0,
          image.cell_counts[1] > 1 ? image.cell_counts[1] : 0,
          image.cell_counts[2] > 1 ? image.cell_counts[2] : 0};
}

Index numCells(const VtiWriter::Image& image)
{
  Index count = 1;
  for (Index cells : image.cell_counts)
  {
    if (cells <= 0)
    {
      throw std::runtime_error("VtiWriter cell counts must be positive");
    }
    if (count > std::numeric_limits<Index>::max() / cells)
    {
      throw std::runtime_error("VtiWriter cell count overflow");
    }
    count *= cells;
  }
  return count;
}

void checkFinite(const VtiWriter::Image& image)
{
  for (Index d = 0; d < 3; ++d)
  {
    if (!std::isfinite(image.origin[static_cast<std::size_t>(d)])
        || !std::isfinite(image.spacing[static_cast<std::size_t>(d)]))
    {
      throw std::runtime_error("VtiWriter origin and spacing must be finite");
    }
  }
  if (image.time && !std::isfinite(*image.time))
  {
    throw std::runtime_error("VtiWriter time must be finite");
  }
}

std::uint64_t fieldBytes(const VtiWriter::CellField& field)
{
  if (field.values == nullptr)
  {
    throw std::runtime_error("VtiWriter cell field has null values");
  }
  const auto size = static_cast<std::uint64_t>(field.values->size());
  if (size > std::numeric_limits<std::uint64_t>::max() / sizeof(Real))
  {
    throw std::runtime_error("VtiWriter cell field is too large");
  }
  return size * static_cast<std::uint64_t>(sizeof(Real));
}

void checkFields(const std::vector<VtiWriter::CellField>& fields,
                 Index                                    cells)
{
  if (fields.empty())
  {
    throw std::runtime_error("VtiWriter needs at least one cell field");
  }

  for (const auto& field : fields)
  {
    if (field.name.empty())
    {
      throw std::runtime_error("VtiWriter cell field name must not be empty");
    }
    if (field.num_components <= 0)
    {
      throw std::runtime_error(
          "VtiWriter cell field component count must be positive");
    }
    if (field.values == nullptr)
    {
      throw std::runtime_error("VtiWriter cell field has null values");
    }
    if (field.values->size() != cells * field.num_components)
    {
      throw std::runtime_error(
          "VtiWriter cell field size does not match image cell count");
    }
  }
}

std::string extentString(const std::array<Index, 3>& max_extent)
{
  std::ostringstream out;
  out << "0 " << max_extent[0] << " 0 " << max_extent[1] << " 0 "
      << max_extent[2];
  return out.str();
}

std::string point3String(const std::array<Real, 3>& values)
{
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<Real>::max_digits10)
      << values[0] << ' ' << values[1] << ' ' << values[2];
  return out.str();
}

} // namespace

void VtiWriter::writeCellData(const std::string&            filename,
                              const Image&                  image,
                              const std::vector<CellField>& fields) const
{
  if (!isLittleEndian())
  {
    throw std::runtime_error(
        "VtiWriter currently writes little-endian VTI files only");
  }

  checkFinite(image);
  const Index cells = numCells(image);
  checkFields(fields, cells);

  std::vector<std::uint64_t> offsets(fields.size(), 0);
  std::uint64_t              offset = 0;
  for (std::size_t i = 0; i < fields.size(); ++i)
  {
    offsets[i] = offset;
    offset += sizeof(std::uint64_t) + fieldBytes(fields[i]);
  }

  std::ofstream out(filename, std::ios::binary);
  if (!out)
  {
    throw std::runtime_error("Failed to open VTI file: " + filename);
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
        << std::setprecision(std::numeric_limits<Real>::max_digits10)
        << *image.time << "</DataArray>\n";
    out << "    </FieldData>\n";
  }
  out << "    <Piece Extent=\"" << extent << "\">\n";
  out << "      <CellData>\n";
  for (std::size_t i = 0; i < fields.size(); ++i)
  {
    const auto& field = fields[i];
    out << "        <DataArray type=\"Float64\" Name=\""
        << escapeXml(field.name) << "\"";
    if (field.num_components > 1)
    {
      out << " NumberOfComponents=\"" << field.num_components << "\"";
    }
    out << " format=\"appended\" offset=\"" << offsets[i] << "\"/>\n";
  }
  out << "      </CellData>\n";
  out << "    </Piece>\n";
  out << "  </ImageData>\n";
  out << "  <AppendedData encoding=\"raw\">\n_";

  for (const auto& field : fields)
  {
    const std::uint64_t bytes = fieldBytes(field);
    out.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
    out.write(reinterpret_cast<const char*>(field.values->data()),
              static_cast<std::streamsize>(bytes));
  }

  out << "\n  </AppendedData>\n";
  out << "</VTKFile>\n";
}

} // namespace femx
