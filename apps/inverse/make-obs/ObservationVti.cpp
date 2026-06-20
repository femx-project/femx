#include "ObservationVti.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <femx/fem/ObservationGrid.hpp>
#include <femx/io/VtiWriter.hpp>
#include <femx/algebra/Vector.hpp>

namespace femx::make_obs
{
namespace
{

using femx::problem::TimeObservationData;

constexpr Real point_tol = 1.0e-10;

struct VtiFields
{
  Vector<Real> velocity;
  Vector<Real> mask;
};

Point3 toPoint(const std::array<Real, 3>& values)
{
  return {values[0], values[1], values[2]};
}

std::string stripKnownExtension(std::string path)
{
  const auto strip = [&path](const std::string& ext)
  {
    if (path.size() >= ext.size()
        && path.compare(path.size() - ext.size(), ext.size(), ext) == 0)
    {
      path.resize(path.size() - ext.size());
      return true;
    }
    return false;
  };

  strip(".vti") || strip(".pvd");
  return path;
}

std::string filenameOnly(const std::string& path)
{
  const std::size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos)
  {
    return path;
  }
  return path.substr(pos + 1);
}

std::string stepTag(Index step)
{
  std::ostringstream out;
  out << std::setw(5) << std::setfill('0') << step;
  return out.str();
}

std::string vtiPath(const std::string& root,
                    Index              level)
{
  return root + "_" + stepTag(level) + ".vti";
}

std::string pvdPath(const std::string& root)
{
  return root + ".pvd";
}

void ensureParentDir(const std::string& path)
{
  const std::filesystem::path parent =
      std::filesystem::path(path).parent_path();
  if (!parent.empty())
  {
    std::filesystem::create_directories(parent);
  }
}

Index checkedGridPointCount(const std::array<Index, 3>& counts)
{
  Index product = 1;
  for (Index count : counts)
  {
    if (count <= 0)
    {
      throw std::runtime_error(
          "make_obs.output.vti requires positive observation grid counts");
    }
    if (product > std::numeric_limits<Index>::max() / count)
    {
      throw std::runtime_error("observation grid count overflow");
    }
    product *= count;
  }
  return product;
}

std::vector<Point3> gridObsPoints(
    const navier_var::ObservationParams::Grid& grid)
{
  if (grid.use_spacing)
  {
    return fem::observationGridPoints(
        toPoint(grid.origin), grid.counts, toPoint(grid.spacing));
  }
  return fem::observationGridPoints(
      toPoint(grid.lower), toPoint(grid.upper), grid.counts);
}

bool samePoint(const Point3& a,
               const Point3& b)
{
  for (Index d = 0; d < 3; ++d)
  {
    const Real scale =
        std::max<Real>({1.0, std::abs(a[d]), std::abs(b[d])});
    if (std::abs(a[d] - b[d]) > point_tol * scale)
    {
      return false;
    }
  }
  return true;
}

VtiWriter::Image makeImage(const std::vector<Point3>& points,
                           const std::array<Index, 3>& counts)
{
  const Index expected_points = checkedGridPointCount(counts);
  if (static_cast<Index>(points.size()) != expected_points)
  {
    throw std::runtime_error(
        "make_obs.output.vti grid counts do not match observation points");
  }
  if (points.empty())
  {
    throw std::runtime_error("make_obs.output.vti received no points");
  }

  VtiWriter::Image image;
  image.cell_counts = counts;

  const Point3&              first  = points.front();
  const std::array<Index, 3> stride = {1, counts[0], counts[0] * counts[1]};
  for (Index d = 0; d < 3; ++d)
  {
    if (counts[static_cast<std::size_t>(d)] > 1)
    {
      const Point3& next    = points[static_cast<std::size_t>(
          stride[static_cast<std::size_t>(d)])];
      const Real    spacing = next[static_cast<std::size_t>(d)]
                           - first[static_cast<std::size_t>(d)];
      if (!std::isfinite(spacing)
          || std::abs(spacing)
                 <= 16.0 * std::numeric_limits<Real>::epsilon())
      {
        throw std::runtime_error(
            "make_obs.output.vti could not infer grid spacing");
      }
      image.spacing[static_cast<std::size_t>(d)] = spacing;
      image.origin[static_cast<std::size_t>(d)] =
          first[static_cast<std::size_t>(d)] - 0.5 * spacing;
    }
    else
    {
      image.spacing[static_cast<std::size_t>(d)] = 1.0;
      image.origin[static_cast<std::size_t>(d)] =
          first[static_cast<std::size_t>(d)];
    }
  }

  return image;
}

Real observationTime(const TimeObservationData& data,
                     Index                      level)
{
  if (data.hasTimeValues())
  {
    return data.timeValue(level);
  }
  return static_cast<Real>(data.timeLevel(level));
}

VtiFields observationCellFields(const TimeObservationData& data,
                                Index                      level,
                                const std::vector<Point3>& grid_points)
{
  const Index num_data_points =
      static_cast<Index>(data.points().size());
  const Index num_components = data.components().size();
  if (num_data_points <= 0 || num_components <= 0)
  {
    throw std::runtime_error(
        "make_obs.output.vti requires observation layout components");
  }

  const Index num_grid_points =
      static_cast<Index>(grid_points.size());
  VtiFields   fields{Vector<Real>(num_grid_points * 3),
                   Vector<Real>(num_grid_points)};
  fields.velocity.setZero();
  fields.mask.setZero();

  const Vector<Real> values = data[level];
  if (values.size() != num_data_points * num_components)
  {
    throw std::runtime_error(
        "make_obs.output.vti data size does not match observation layout");
  }

  Index data_point = 0;
  for (Index grid_point = 0; grid_point < num_grid_points; ++grid_point)
  {
    if (data_point >= num_data_points
        || !samePoint(grid_points[static_cast<std::size_t>(grid_point)],
                      data.points()[static_cast<std::size_t>(data_point)]))
    {
      continue;
    }

    fields.mask[grid_point] = 1.0;
    for (Index local_component = 0; local_component < num_components;
         ++local_component)
    {
      const Index component = data.components()[local_component];
      if (component < 0 || component >= 3)
      {
        throw std::runtime_error(
            "make_obs.output.vti supports velocity components 0, 1, and 2");
      }
      fields.velocity[3 * grid_point + component] =
          values[data_point * num_components + local_component];
    }
    ++data_point;
  }
  if (data_point != num_data_points)
  {
    throw std::runtime_error(
        "make_obs.output.vti observation points are not an ordered grid subset");
  }
  return fields;
}

void writePvd(const std::string&         path,
              const std::string&         root,
              const TimeObservationData& data)
{
  ensureParentDir(path);
  std::ofstream out(path);
  if (!out)
  {
    throw std::runtime_error("Failed to open observation PVD file: " + path);
  }

  out << std::setprecision(std::numeric_limits<Real>::max_digits10);
  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"Collection\" version=\"1.0\" "
         "byte_order=\"LittleEndian\">\n";
  out << "  <Collection>\n";
  for (Index level = 0; level < data.numLevels(); ++level)
  {
    out << "    <DataSet timestep=\"" << observationTime(data, level)
        << "\" group=\"\" part=\"0\" file=\""
        << filenameOnly(vtiPath(root, level)) << "\"/>\n";
  }
  out << "  </Collection>\n";
  out << "</VTKFile>\n";
}

} // namespace

std::string writeObservationVtiOutputs(const Params&              prm,
                                       const TimeObservationData& data)
{
  if (!prm.output.write_vti)
  {
    return {};
  }
  if (!prm.obs.grid)
  {
    throw std::runtime_error(
        "make_obs.output.vti requires make_obs.obs.grid");
  }
  if (!data.hasLayout())
  {
    throw std::runtime_error(
        "make_obs.output.vti requires observation point layout");
  }

  const std::string root = stripKnownExtension(prm.output.vti_basename);
  if (root.empty())
  {
    throw std::runtime_error("make_obs.output.vti_basename must not be empty");
  }

  const std::vector<Point3> grid_points = gridObsPoints(*prm.obs.grid);

  VtiWriter writer;
  for (Index level = 0; level < data.numLevels(); ++level)
  {
    VtiWriter::Image image =
        makeImage(grid_points, prm.obs.grid->counts);
    VtiFields fields = observationCellFields(data, level, grid_points);

    const std::string file = vtiPath(root, level);
    ensureParentDir(file);
    writer.writeCellData(file,
                         image,
                         {{"mask", 1, &fields.mask},
                          {"velocity", 3, &fields.velocity}});
  }

  const std::string pvd = pvdPath(root);
  writePvd(pvd, root, data);
  return pvd;
}

} // namespace femx::make_obs
