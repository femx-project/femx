#include <femx/io/TimeSeriesDataIn.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string>

#ifdef FEMX_HAS_HDF5
#include <hdf5.h>
#endif

namespace femx
{
namespace
{

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

  strip(".xdmf") || strip(".h5");
  return path;
}

std::string stepName(Index step)
{
  std::string tag = std::to_string(step);
  while (tag.size() < 5)
  {
    tag = "0" + tag;
  }
  return "Step" + tag;
}

struct SeriesPaths
{
  std::string root;
  std::string hdf5;
  std::string xdmf;
};

SeriesPaths seriesPaths(const std::string& path)
{
  const std::string root = stripKnownExtension(path);
  return {root, root + ".h5", root + ".xdmf"};
}

Cell::Shape shapeFromName(const std::string& name)
{
  if (name == "Triangle")
  {
    return Cell::Shape::Triangle;
  }
  if (name == "Quadrilateral")
  {
    return Cell::Shape::Quadrilateral;
  }
  if (name == "Tetrahedron")
  {
    return Cell::Shape::Tetrahedron;
  }
  return Cell::Shape::Unknown;
}

Cell::Shape inferShape(Index cell_nodes,
                       Index mesh_dim)
{
  if (cell_nodes == 3 && mesh_dim == 2)
  {
    return Cell::Shape::Triangle;
  }
  if (cell_nodes == 4 && mesh_dim == 2)
  {
    return Cell::Shape::Quadrilateral;
  }
  if (cell_nodes == 4 && mesh_dim == 3)
  {
    return Cell::Shape::Tetrahedron;
  }
  throw std::runtime_error("TimeSeriesDataIn cannot infer mesh cell shape");
}

struct XdmfInfo
{
  std::vector<Real> times;
  std::vector<std::string> vector_fields;
  Cell::Shape       shape = Cell::Shape::Unknown;
};

std::string xmlAttribute(const std::string& text,
                         const std::string& name)
{
  const std::regex re(name + "=\"([^\"]+)\"");
  std::smatch      match;
  if (std::regex_search(text, match, re))
  {
    return match[1].str();
  }
  return {};
}

void addUnique(std::vector<std::string>& values,
               const std::string&        value)
{
  for (const std::string& existing : values)
  {
    if (existing == value)
    {
      return;
    }
  }
  values.push_back(value);
}

XdmfInfo readXdmfInfo(const std::string& path)
{
  XdmfInfo info;
  std::ifstream in(path);
  if (!in)
  {
    return info;
  }

  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());

  const std::regex time_re("<Time\\s+Value=\"([^\"]+)\"");
  for (std::sregex_iterator it(text.begin(), text.end(), time_re), end;
       it != end;
       ++it)
  {
    info.times.push_back(std::stod((*it)[1].str()));
  }

  const std::regex topology_re("TopologyType=\"([^\"]+)\"");
  std::smatch      match;
  if (std::regex_search(text, match, topology_re))
  {
    info.shape = shapeFromName(match[1].str());
  }

  const std::regex attribute_re("<Attribute\\s+([^>]*)>");
  for (std::sregex_iterator it(text.begin(), text.end(), attribute_re), end;
       it != end;
       ++it)
  {
    const std::string tag  = (*it)[1].str();
    const std::string type = xmlAttribute(tag, "AttributeType");
    const std::string name = xmlAttribute(tag, "Name");
    if (type == "Vector" && !name.empty())
    {
      addUnique(info.vector_fields, name);
    }
  }
  return info;
}

Index meshDim(const std::vector<double>& geometry,
              Cell::Shape                shape)
{
  if (shape == Cell::Shape::Tetrahedron)
  {
    return 3;
  }

  bool flat_z = true;
  for (std::size_t i = 2; i < geometry.size(); i += 3)
  {
    if (std::abs(geometry[i]) > 1.0e-14)
    {
      flat_z = false;
      break;
    }
  }
  return flat_z ? 2 : 3;
}

#ifdef FEMX_HAS_HDF5

void checkHdf5(herr_t status, const std::string& message)
{
  if (status < 0)
  {
    throw std::runtime_error(message);
  }
}

void checkHdf5Id(hid_t id, const std::string& message)
{
  if (id < 0)
  {
    throw std::runtime_error(message);
  }
}

bool linkExists(hid_t file, const std::string& path)
{
  return H5Lexists(file, path.c_str(), H5P_DEFAULT) > 0;
}

std::vector<hsize_t> datasetDims(hid_t dataset,
                                 const std::string& path)
{
  hid_t space = H5Dget_space(dataset);
  checkHdf5Id(space, "Failed to get dataspace for " + path);
  const int ndims = H5Sget_simple_extent_ndims(space);
  if (ndims <= 0)
  {
    checkHdf5(H5Sclose(space), "Failed to close dataspace for " + path);
    throw std::runtime_error("Dataset has invalid rank: " + path);
  }

  std::vector<hsize_t> dims(static_cast<std::size_t>(ndims));
  checkHdf5(H5Sget_simple_extent_dims(space, dims.data(), nullptr),
            "Failed to read dimensions for " + path);
  checkHdf5(H5Sclose(space), "Failed to close dataspace for " + path);
  return dims;
}

std::vector<double> readDoubleDataset(hid_t              file,
                                      const std::string& path,
                                      std::vector<hsize_t>& dims)
{
  hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  checkHdf5Id(dataset, "Failed to open dataset " + path);
  dims = datasetDims(dataset, path);

  hsize_t count = 1;
  for (hsize_t dim : dims)
  {
    count *= dim;
  }
  std::vector<double> data(static_cast<std::size_t>(count));
  checkHdf5(H5Dread(dataset,
                    H5T_NATIVE_DOUBLE,
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    data.data()),
            "Failed to read dataset " + path);
  checkHdf5(H5Dclose(dataset), "Failed to close dataset " + path);
  return data;
}

std::vector<Index> readIntDataset(hid_t              file,
                                  const std::string& path,
                                  std::vector<hsize_t>& dims)
{
  hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  checkHdf5Id(dataset, "Failed to open dataset " + path);
  dims = datasetDims(dataset, path);

  hsize_t count = 1;
  for (hsize_t dim : dims)
  {
    count *= dim;
  }
  std::vector<Index> data(static_cast<std::size_t>(count));
  checkHdf5(H5Dread(dataset,
                    H5T_NATIVE_INT,
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    data.data()),
            "Failed to read dataset " + path);
  checkHdf5(H5Dclose(dataset), "Failed to close dataset " + path);
  return data;
}

Mesh readMesh(hid_t file, Cell::Shape shape)
{
  std::vector<hsize_t> geom_dims;
  const auto geometry =
      readDoubleDataset(file, "/Mesh/Geometry", geom_dims);
  if (geom_dims.size() != 2 || geom_dims[1] != 3)
  {
    throw std::runtime_error("TimeSeriesDataIn expects /Mesh/Geometry as N x 3");
  }

  std::vector<hsize_t> topo_dims;
  const auto topology =
      readIntDataset(file, "/Mesh/Topology", topo_dims);
  if (topo_dims.size() != 2)
  {
    throw std::runtime_error("TimeSeriesDataIn expects /Mesh/Topology as E x K");
  }

  const Index nodes      = static_cast<Index>(geom_dims[0]);
  const Index elems      = static_cast<Index>(topo_dims[0]);
  const Index cell_nodes = static_cast<Index>(topo_dims[1]);
  const Index dim        = meshDim(geometry, shape);
  if (shape == Cell::Shape::Unknown)
  {
    shape = inferShape(cell_nodes, dim);
  }

  Mesh mesh(dim);
  for (Index in = 0; in < nodes; ++in)
  {
    mesh.addNode({geometry[static_cast<std::size_t>(3 * in)],
                  geometry[static_cast<std::size_t>(3 * in + 1)],
                  geometry[static_cast<std::size_t>(3 * in + 2)]});
  }

  for (Index ic = 0; ic < elems; ++ic)
  {
    std::vector<Index> node_ids;
    node_ids.reserve(static_cast<std::size_t>(cell_nodes));
    for (Index i = 0; i < cell_nodes; ++i)
    {
      node_ids.push_back(
          topology[static_cast<std::size_t>(ic * cell_nodes + i)]);
    }
    mesh.addCell(node_ids, shape, dim, 0, 0, {});
  }
  return mesh;
}

std::array<Vector<Real>, 3> readVectorField(hid_t              file,
                                            const std::string& path,
                                            Index              nodes)
{
  std::vector<hsize_t> dims;
  const auto data = readDoubleDataset(file, path, dims);
  if (dims.size() != 2 || dims[0] != static_cast<hsize_t>(nodes)
      || dims[1] != 3)
  {
    throw std::runtime_error(
        "TimeSeriesDataIn expects vector field as num_nodes x 3: " + path);
  }

  std::array<Vector<Real>, 3> out{
      Vector<Real>(nodes), Vector<Real>(nodes), Vector<Real>(nodes)};
  for (Index in = 0; in < nodes; ++in)
  {
    for (Index d = 0; d < 3; ++d)
    {
      out[static_cast<std::size_t>(d)][in] =
          data[static_cast<std::size_t>(3 * in + d)];
    }
  }
  return out;
}

Index countSteps(hid_t file)
{
  Index count = 0;
  while (linkExists(file, "/Data/" + stepName(count)))
  {
    ++count;
  }
  return count;
}

#endif

} // namespace

TimeSeriesDataIn TimeSeriesDataIn::read(const std::string& path)
{
#ifndef FEMX_HAS_HDF5
  (void) path;
  throw std::runtime_error(
      "HDF5 support is not enabled. Configure with FEMX_ENABLE_HDF5=ON.");
#else
  const SeriesPaths paths = seriesPaths(path);
  const XdmfInfo    xdmf  = readXdmfInfo(paths.xdmf);

  hid_t file = H5Fopen(paths.hdf5.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  checkHdf5Id(file, "Failed to open HDF5 time series: " + paths.hdf5);

  TimeSeriesDataIn out;
  out.mesh_ = readMesh(file, xdmf.shape);

  const Index hdf5_steps = countSteps(file);
  if (hdf5_steps <= 0)
  {
    checkHdf5(H5Fclose(file), "Failed to close HDF5 time series");
    throw std::runtime_error("TimeSeriesDataIn found no /Data/Stepxxxxx groups");
  }
  if (!xdmf.times.empty()
      && static_cast<Index>(xdmf.times.size()) != hdf5_steps)
  {
    checkHdf5(H5Fclose(file), "Failed to close HDF5 time series");
    throw std::runtime_error("XDMF time count does not match HDF5 steps");
  }

  out.steps_.resize(static_cast<std::size_t>(hdf5_steps));
  std::vector<std::string> vector_fields = xdmf.vector_fields;
  if (vector_fields.empty())
  {
    vector_fields.push_back("velocity");
  }

  for (Index step = 0; step < hdf5_steps; ++step)
  {
    Step& dst = out.steps_[static_cast<std::size_t>(step)];
    dst.time =
        xdmf.times.empty() ? static_cast<Real>(step)
                           : xdmf.times[static_cast<std::size_t>(step)];

    const std::string group = "/Data/" + stepName(step);
    for (const std::string& field : vector_fields)
    {
      const std::string field_path = group + "/" + field;
      if (linkExists(file, field_path))
      {
        dst.vecs.push_back(
            {field, readVectorField(file, field_path, out.mesh_.numNodes())});
      }
    }
  }

  checkHdf5(H5Fclose(file), "Failed to close HDF5 time series");
  return out;
#endif
}

const Mesh& TimeSeriesDataIn::mesh() const
{
  return mesh_;
}

Index TimeSeriesDataIn::numSteps() const
{
  return static_cast<Index>(steps_.size());
}

Real TimeSeriesDataIn::time(Index step) const
{
  return steps_[static_cast<std::size_t>(step)].time;
}

const std::array<Vector<Real>, 3>& TimeSeriesDataIn::vectorField(
    Index              step,
    const std::string& name) const
{
  const Step& data = steps_[static_cast<std::size_t>(step)];
  for (const VectorField& field : data.vecs)
  {
    if (field.name == name)
    {
      return field.values;
    }
  }
  throw std::runtime_error(
      "TimeSeriesDataIn missing vector field '" + name + "'");
}

} // namespace femx
