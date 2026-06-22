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

#include <femx/fem/ObservationGrid.hpp>
#include <femx/io/VtiWriter.hpp>
#include <femx/linalg/Vector.hpp>

using namespace std;
using namespace femx;
using namespace femx::problem;
using namespace femx::fem;

namespace femx::make_obs
{
namespace
{

constexpr Real point_tol = 1.0e-10;

struct VtiFields
{
  Vector<Real> velocity;
  Vector<Real> mask;
};

Point3 toPoint(const array<Real, 3>& vals)
{
  return {vals[0], vals[1], vals[2]};
}

string stripKnownExtension(string path)
{
  const auto strip = [&path](const string& ext)
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

string filenameOnly(const string& path)
{
  const size_t pos = path.find_last_of("/\\");
  if (pos == string::npos)
  {
    return path;
  }
  return path.substr(pos + 1);
}

string stepTag(Index step)
{
  ostringstream out;
  out << setw(5) << setfill('0') << step;
  return out.str();
}

string vtiPath(const string& root,
               Index         level)
{
  return root + "_" + stepTag(level) + ".vti";
}

string pvdPath(const string& root)
{
  return root + ".pvd";
}

void ensureParentDir(const string& path)
{
  const filesystem::path parent =
      filesystem::path(path).parent_path();
  if (!parent.empty())
  {
    filesystem::create_directories(parent);
  }
}

Index checkedGridPointCount(const array<Index, 3>& counts)
{
  Index product = 1;
  for (Index count : counts)
  {
    if (count <= 0)
    {
      throw runtime_error(
          "make_obs.output.vti requires positive observation grid counts");
    }
    if (product > numeric_limits<Index>::max() / count)
    {
      throw runtime_error("observation grid count overflow");
    }
    product *= count;
  }
  return product;
}

Vector<Point3> gridObsPoints(
    const navier_var_new::ObservationParams::Grid& grid)
{
  if (grid.use_spacing)
  {
    return observationGridPoints(
        toPoint(grid.origin), grid.counts, toPoint(grid.spacing));
  }
  return observationGridPoints(
      toPoint(grid.lower), toPoint(grid.upper), grid.counts);
}

bool samePoint(const Point3& a,
               const Point3& b)
{
  for (Index d = 0; d < 3; ++d)
  {
    const Real scale =
        max<Real>({1.0, abs(a[d]), abs(b[d])});
    if (abs(a[d] - b[d]) > point_tol * scale)
    {
      return false;
    }
  }
  return true;
}

VtiWriter::Image makeImage(const Vector<Point3>&  pts,
                           const array<Index, 3>& counts)
{
  const Index expected_points = checkedGridPointCount(counts);
  if (pts.size() != expected_points)
  {
    throw runtime_error(
        "make_obs.output.vti grid counts do not match observation points");
  }
  if (pts.empty())
  {
    throw runtime_error("make_obs.output.vti received no points");
  }

  VtiWriter::Image image;
  image.cell_counts = counts;

  const Point3&         first  = pts.front();
  const array<Index, 3> stride = {1, counts[0], counts[0] * counts[1]};
  for (Index d = 0; d < 3; ++d)
  {
    if (counts[d] > 1)
    {
      const Point3& next    = pts[stride[d]];
      const Real    spacing = next[d] - first[d];
      if (!isfinite(spacing)
          || abs(spacing)
                 <= 16.0 * numeric_limits<Real>::epsilon())
      {
        throw runtime_error(
            "make_obs.output.vti could not infer grid spacing");
      }
      image.spacing[d] = spacing;
      image.origin[d]  = first[d] - 0.5 * spacing;
    }
    else
    {
      image.spacing[d] = 1.0;
      image.origin[d]  = first[d];
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
                                const Vector<Point3>&      grid_points)
{
  const Index num_data_points = data.pts().size();
  const Index nc              = data.comps().size();
  if (num_data_points <= 0 || nc <= 0)
  {
    throw runtime_error(
        "make_obs.output.vti requires observation layout components");
  }

  const Index num_grid_points = grid_points.size();
  VtiFields   fields{Vector<Real>(num_grid_points * 3),
                   Vector<Real>(num_grid_points)};
  fields.velocity.setZero();
  fields.mask.setZero();

  const Vector<Real> vals = data[level];
  if (vals.size() != num_data_points * nc)
  {
    throw runtime_error(
        "make_obs.output.vti data size does not match observation layout");
  }

  Index data_point = 0;
  for (Index grid_point = 0; grid_point < num_grid_points; ++grid_point)
  {
    if (data_point >= num_data_points
        || !samePoint(grid_points[grid_point],
                      data.pts()[data_point]))
    {
      continue;
    }

    fields.mask[grid_point] = 1.0;
    for (Index local_component = 0; local_component < nc;
         ++local_component)
    {
      const Index comp = data.comps()[local_component];
      if (comp < 0 || comp >= 3)
      {
        throw runtime_error(
            "make_obs.output.vti supports velocity components 0, 1, and 2");
      }
      fields.velocity[3 * grid_point + comp] =
          vals[data_point * nc + local_component];
    }
    ++data_point;
  }
  if (data_point != num_data_points)
  {
    throw runtime_error(
        "make_obs.output.vti observation points are not an ordered grid subset");
  }
  return fields;
}

void writePvd(const string&              path,
              const string&              root,
              const TimeObservationData& data)
{
  ensureParentDir(path);
  ofstream out(path);
  if (!out)
  {
    throw runtime_error("Failed to open observation PVD file: " + path);
  }

  out << setprecision(numeric_limits<Real>::max_digits10);
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

string writeObservationVtiOutputs(const Params&              prm,
                                  const TimeObservationData& data)
{
  if (!prm.output.write_vti)
  {
    return {};
  }
  if (!prm.obs.grid)
  {
    throw runtime_error(
        "make_obs.output.vti requires make_obs.obs.grid");
  }
  if (!data.hasLayout())
  {
    throw runtime_error(
        "make_obs.output.vti requires observation point layout");
  }

  const string root = stripKnownExtension(prm.output.vti_basename);
  if (root.empty())
  {
    throw runtime_error("make_obs.output.vti_basename must not be empty");
  }

  const Vector<Point3> grid_points = gridObsPoints(*prm.obs.grid);

  VtiWriter writer;
  for (Index level = 0; level < data.numLevels(); ++level)
  {
    VtiWriter::Image image =
        makeImage(grid_points, prm.obs.grid->counts);
    VtiFields fields = observationCellFields(data, level, grid_points);

    const string file = vtiPath(root, level);
    ensureParentDir(file);
    writer.writeCellData(file,
                         image,
                         {{"mask", 1, &fields.mask},
                          {"velocity", 3, &fields.velocity}});
  }

  const string pvd = pvdPath(root);
  writePvd(pvd, root, data);
  return pvd;
}

} // namespace femx::make_obs
