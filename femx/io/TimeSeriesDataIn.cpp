#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string>

#include <femx/io/TimeSeriesDataIn.hpp>

#ifdef FEMX_HAS_HDF5
#include <hdf5.h>
#endif

using namespace std;

namespace femx
{
namespace
{

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

  strip(".xdmf") || strip(".h5");
  return path;
}

string stepName(Index step)
{
  string tag = to_string(step);
  while (tag.size() < 5)
  {
    tag = "0" + tag;
  }
  return "Step" + tag;
}

struct SeriesPaths
{
  string root;
  string hdf5;
  string xdmf;
};

SeriesPaths seriesPaths(const string& path)
{
  const string root = stripKnownExtension(path);
  return {root, root + ".h5", root + ".xdmf"};
}

Cell::Shape shapeFromName(const string& name)
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

Cell::Shape inferShape(Index cn,
                       Index mesh_dim)
{
  if (cn == 3 && mesh_dim == 2)
  {
    return Cell::Shape::Triangle;
  }
  if (cn == 4 && mesh_dim == 2)
  {
    return Cell::Shape::Quadrilateral;
  }
  if (cn == 4 && mesh_dim == 3)
  {
    return Cell::Shape::Tetrahedron;
  }
  throw runtime_error("TimeSeriesDataIn cannot infer mesh cell shape");
}

struct XdmfInfo
{
  Vector<Real>   times;
  Vector<string> vector_fields;
  Cell::Shape    shape = Cell::Shape::Unknown;
};

string xmlAttribute(const string& text,
                    const string& name)
{
  const regex re(name + "=\"([^\"]+)\"");
  smatch      match;
  if (regex_search(text, match, re))
  {
    return match[1].str();
  }
  return {};
}

void addUnique(Vector<string>& vals,
               const string&   value)
{
  for (const string& existing : vals)
  {
    if (existing == value)
    {
      return;
    }
  }
  vals.push_back(value);
}

XdmfInfo readXdmfInfo(const string& path)
{
  XdmfInfo info;
  ifstream in(path);
  if (!in)
  {
    return info;
  }

  const string text((istreambuf_iterator<char>(in)),
                    istreambuf_iterator<char>());

  const regex time_re("<Time\\s+Value=\"([^\"]+)\"");
  for (sregex_iterator it(text.begin(), text.end(), time_re), end;
       it != end;
       ++it)
  {
    info.times.push_back(stod((*it)[1].str()));
  }

  const regex topology_re("TopologyType=\"([^\"]+)\"");
  smatch      match;
  if (regex_search(text, match, topology_re))
  {
    info.shape = shapeFromName(match[1].str());
  }

  const regex attribute_re("<Attribute\\s+([^>]*)>");
  for (sregex_iterator it(text.begin(), text.end(), attribute_re), end;
       it != end;
       ++it)
  {
    const string tag  = (*it)[1].str();
    const string type = xmlAttribute(tag, "AttributeType");
    const string name = xmlAttribute(tag, "Name");
    if (type == "Vector" && !name.empty())
    {
      addUnique(info.vector_fields, name);
    }
  }
  return info;
}

Index meshDim(const Vector<double>& geometry,
              Cell::Shape           shape)
{
  if (shape == Cell::Shape::Tetrahedron)
  {
    return 3;
  }

  bool flat_z = true;
  for (Index i = 2; i < geometry.size(); i += 3)
  {
    if (abs(geometry[i]) > 1.0e-14)
    {
      flat_z = false;
      break;
    }
  }
  return flat_z ? 2 : 3;
}

#ifdef FEMX_HAS_HDF5

void checkHdf5(herr_t status, const string& msg)
{
  if (status < 0)
  {
    throw runtime_error(msg);
  }
}

void checkHdf5Id(hid_t id, const string& msg)
{
  if (id < 0)
  {
    throw runtime_error(msg);
  }
}

bool linkExists(hid_t file, const string& path)
{
  return H5Lexists(file, path.c_str(), H5P_DEFAULT) > 0;
}

Vector<hsize_t> datasetDims(hid_t         dset,
                            const string& path)
{
  hid_t space = H5Dget_space(dset);
  checkHdf5Id(space, "Failed to get dataspace for " + path);
  const int ndims = H5Sget_simple_extent_ndims(space);
  if (ndims <= 0)
  {
    checkHdf5(H5Sclose(space), "Failed to close dataspace for " + path);
    throw runtime_error("Dataset has invalid rank: " + path);
  }

  Vector<hsize_t> dims(ndims);
  checkHdf5(H5Sget_simple_extent_dims(space, dims.data(), nullptr),
            "Failed to read dimensions for " + path);
  checkHdf5(H5Sclose(space), "Failed to close dataspace for " + path);
  return dims;
}

Vector<double> readDoubleDataset(hid_t            file,
                                 const string&    path,
                                 Vector<hsize_t>& dims)
{
  hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  checkHdf5Id(dset, "Failed to open dataset " + path);
  dims = datasetDims(dset, path);

  hsize_t count = 1;
  for (hsize_t dim : dims)
  {
    count *= dim;
  }
  Vector<double> data(static_cast<Index>(count));
  checkHdf5(H5Dread(dset,
                    H5T_NATIVE_DOUBLE,
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    data.data()),
            "Failed to read dataset " + path);
  checkHdf5(H5Dclose(dset), "Failed to close dataset " + path);
  return data;
}

Vector<Index> readIntDataset(hid_t            file,
                             const string&    path,
                             Vector<hsize_t>& dims)
{
  hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  checkHdf5Id(dset, "Failed to open dataset " + path);
  dims = datasetDims(dset, path);

  hsize_t count = 1;
  for (hsize_t dim : dims)
  {
    count *= dim;
  }
  Vector<Index> data(static_cast<Index>(count));
  checkHdf5(H5Dread(dset,
                    H5T_NATIVE_INT,
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    data.data()),
            "Failed to read dataset " + path);
  checkHdf5(H5Dclose(dset), "Failed to close dataset " + path);
  return data;
}

Mesh readMesh(hid_t file, Cell::Shape shape)
{
  Vector<hsize_t> geom_dims;
  const auto      geometry =
      readDoubleDataset(file, "/Mesh/Geometry", geom_dims);
  if (geom_dims.size() != 2 || geom_dims[1] != 3)
  {
    throw runtime_error("TimeSeriesDataIn expects /Mesh/Geometry as N x 3");
  }

  Vector<hsize_t> topo_dims;
  const auto      topology =
      readIntDataset(file, "/Mesh/Topology", topo_dims);
  if (topo_dims.size() != 2)
  {
    throw runtime_error("TimeSeriesDataIn expects /Mesh/Topology as E x K");
  }

  const Index nodes = static_cast<Index>(geom_dims[0]);
  const Index elems = static_cast<Index>(topo_dims[0]);
  const Index cn    = static_cast<Index>(topo_dims[1]);
  const Index dim   = meshDim(geometry, shape);
  if (shape == Cell::Shape::Unknown)
  {
    shape = inferShape(cn, dim);
  }

  Mesh mesh(dim);
  for (Index in = 0; in < nodes; ++in)
  {
    mesh.addNode({geometry[3 * in],
                  geometry[3 * in + 1],
                  geometry[3 * in + 2]});
  }

  for (Index ic = 0; ic < elems; ++ic)
  {
    Vector<Index> nids;
    nids.reserve(cn);
    for (Index i = 0; i < cn; ++i)
    {
      nids.push_back(topology[ic * cn + i]);
    }
    mesh.addCell(nids, shape, dim, 0, 0, {});
  }
  return mesh;
}

array<Vector<Real>, 3> readVectorField(hid_t         file,
                                       const string& path,
                                       Index         nodes)
{
  Vector<hsize_t> dims;
  const auto      data = readDoubleDataset(file, path, dims);
  if (dims.size() != 2 || dims[0] != static_cast<hsize_t>(nodes)
      || dims[1] != 3)
  {
    throw runtime_error(
        "TimeSeriesDataIn expects vector field as num_nodes x 3: " + path);
  }

  array<Vector<Real>, 3> out{
      Vector<Real>(nodes), Vector<Real>(nodes), Vector<Real>(nodes)};
  for (Index in = 0; in < nodes; ++in)
  {
    for (Index d = 0; d < 3; ++d)
    {
      out[d][in] = data[3 * in + d];
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

TimeSeriesDataIn TimeSeriesDataIn::read(const string& path)
{
#ifndef FEMX_HAS_HDF5
  (void) path;
  throw runtime_error(
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
    throw runtime_error("TimeSeriesDataIn found no /Data/Stepxxxxx groups");
  }
  if (!xdmf.times.empty()
      && xdmf.times.size() != hdf5_steps)
  {
    checkHdf5(H5Fclose(file), "Failed to close HDF5 time series");
    throw runtime_error("XDMF time count does not match HDF5 steps");
  }

  out.steps_.resize(hdf5_steps);
  Vector<string> vector_fields = xdmf.vector_fields;
  if (vector_fields.empty())
  {
    vector_fields.push_back("velocity");
  }

  for (Index step = 0; step < hdf5_steps; ++step)
  {
    Step& dst = out.steps_[step];
    dst.time =
        xdmf.times.empty() ? static_cast<Real>(step)
                           : xdmf.times[step];

    const string group = "/Data/" + stepName(step);
    for (const string& field : vector_fields)
    {
      const string field_path = group + "/" + field;
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
  return steps_.size();
}

Real TimeSeriesDataIn::time(Index step) const
{
  return steps_[step].time;
}

const array<Vector<Real>, 3>& TimeSeriesDataIn::vectorField(
    Index         step,
    const string& name) const
{
  const Step& data = steps_[step];
  for (const VectorField& field : data.vecs)
  {
    if (field.name == name)
    {
      return field.vals;
    }
  }
  throw runtime_error(
      "TimeSeriesDataIn missing vector field '" + name + "'");
}

} // namespace femx
