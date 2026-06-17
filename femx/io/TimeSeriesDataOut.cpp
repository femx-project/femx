#include <fstream>
#include <stdexcept>
#include <string>

#include <femx/io/TimeSeriesDataOut.hpp>
#include <femx/mesh/Cell.hpp>
#include <femx/mesh/Mesh.hpp>

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
    if (path.size() >= ext.size() && path.compare(path.size() - ext.size(), ext.size(), ext) == 0)
    {
      path.resize(path.size() - ext.size());
      return true;
    }
    return false;
  };

  strip(".xdmf") || strip(".h5");
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

std::string stepName(Index step)
{
  std::string tag = std::to_string(step);
  while (tag.size() < 5)
  {
    tag = "0" + tag;
  }
  return "Step" + tag;
}

void checkMesh(const Mesh& mesh)
{
  if (mesh.dim() != 2 && mesh.dim() != 3)
  {
    throw std::runtime_error("TimeSeriesDataOut supports 2D and 3D meshes");
  }

  if (mesh.numElems() == 0)
  {
    throw std::runtime_error("TimeSeriesDataOut needs a non-empty mesh");
  }

  const Index       cell_nodes = mesh.cells().front().numNodes();
  const Cell::Shape shape      = mesh.cells().front().shape();
  if (shape != Cell::Shape::Triangle && shape != Cell::Shape::Quadrilateral && shape != Cell::Shape::Tetrahedron)
  {
    throw std::runtime_error(
        "TimeSeriesDataOut supports triangle, quadrilateral, and tetrahedron cells");
  }

  for (Index ic = 1; ic < mesh.numElems(); ++ic)
  {
    const auto& cell = mesh.cell(ic);
    if (cell.numNodes() != cell_nodes || cell.shape() != shape)
    {
      throw std::runtime_error("TimeSeriesDataOut supports one cell type per mesh");
    }
  }
}

void checkFieldSize(const Mesh& mesh, const Vector<Real>& values)
{
  if (values.size() != mesh.numNodes())
  {
    throw std::runtime_error(
        "TimeSeriesDataOut expects one field value per mesh node");
  }
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

void writeDoubleDataset(hid_t                       file,
                        const std::string&          path,
                        const std::vector<double>&  data,
                        const std::vector<hsize_t>& dims)
{
  hid_t dataspace =
      H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
  checkHdf5Id(dataspace, "Failed to create HDF5 dataspace for " + path);

  hid_t dataset = H5Dcreate2(file,
                             path.c_str(),
                             H5T_NATIVE_DOUBLE,
                             dataspace,
                             H5P_DEFAULT,
                             H5P_DEFAULT,
                             H5P_DEFAULT);
  checkHdf5Id(dataset, "Failed to create HDF5 dataset " + path);

  checkHdf5(H5Dwrite(dataset,
                     H5T_NATIVE_DOUBLE,
                     H5S_ALL,
                     H5S_ALL,
                     H5P_DEFAULT,
                     data.data()),
            "Failed to write HDF5 dataset " + path);

  checkHdf5(H5Dclose(dataset), "Failed to close HDF5 dataset " + path);
  checkHdf5(H5Sclose(dataspace), "Failed to close HDF5 dataspace " + path);
}

void writeIntDataset(hid_t                       file,
                     const std::string&          path,
                     const std::vector<Index>&   data,
                     const std::vector<hsize_t>& dims)
{
  hid_t dataspace =
      H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
  checkHdf5Id(dataspace, "Failed to create HDF5 dataspace for " + path);

  hid_t dataset = H5Dcreate2(file,
                             path.c_str(),
                             H5T_NATIVE_INT,
                             dataspace,
                             H5P_DEFAULT,
                             H5P_DEFAULT,
                             H5P_DEFAULT);
  checkHdf5Id(dataset, "Failed to create HDF5 dataset " + path);

  checkHdf5(H5Dwrite(dataset,
                     H5T_NATIVE_INT,
                     H5S_ALL,
                     H5S_ALL,
                     H5P_DEFAULT,
                     data.data()),
            "Failed to write HDF5 dataset " + path);

  checkHdf5(H5Dclose(dataset), "Failed to close HDF5 dataset " + path);
  checkHdf5(H5Sclose(dataspace), "Failed to close HDF5 dataspace " + path);
}

void writeScalarDataset(hid_t               file,
                        const std::string&  path,
                        const Vector<Real>& values)
{
  std::vector<double> data(static_cast<std::size_t>(values.size()));
  for (Index i = 0; i < values.size(); ++i)
  {
    data[static_cast<std::size_t>(i)] = values[i];
  }

  writeDoubleDataset(file,
                     path,
                     data,
                     {static_cast<hsize_t>(values.size())});
}

void writeVectorDataset(hid_t                              file,
                        const std::string&                 path,
                        const std::array<Vector<Real>, 3>& values)
{
  const Index         nodes = values[0].size();
  std::vector<double> data(static_cast<std::size_t>(nodes) * 3);

  for (Index in = 0; in < nodes; ++in)
  {
    for (Index d = 0; d < 3; ++d)
    {
      data[static_cast<std::size_t>(in) * 3 + static_cast<std::size_t>(d)] = values[d][in];
    }
  }

  writeDoubleDataset(file,
                     path,
                     data,
                     {static_cast<hsize_t>(nodes), 3});
}

void writeMesh(hid_t file, const Mesh& mesh)
{
  hid_t mesh_group =
      H5Gcreate2(file, "/Mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  checkHdf5Id(mesh_group, "Failed to create /Mesh group");
  checkHdf5(H5Gclose(mesh_group), "Failed to close /Mesh group");

  hid_t data_group =
      H5Gcreate2(file, "/Data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  checkHdf5Id(data_group, "Failed to create /Data group");
  checkHdf5(H5Gclose(data_group), "Failed to close /Data group");

  std::vector<double> geometry(static_cast<std::size_t>(mesh.numNodes()) * 3);
  for (Index in = 0; in < mesh.numNodes(); ++in)
  {
    for (Index d = 0; d < 3; ++d)
    {
      geometry[static_cast<std::size_t>(in) * 3 + static_cast<std::size_t>(d)] = mesh.node(in)[d];
    }
  }

  if (mesh.numElems() == 0)
  {
    throw std::runtime_error("TimeSeriesDataOut needs a non-empty mesh");
  }

  const Index       cell_nodes = mesh.cells().front().numNodes();
  const Cell::Shape shape      = mesh.cells().front().shape();
  if (shape != Cell::Shape::Triangle && shape != Cell::Shape::Quadrilateral && shape != Cell::Shape::Tetrahedron)
  {
    throw std::runtime_error(
        "TimeSeriesDataOut supports triangle, quadrilateral, and tetrahedron cells");
  }

  std::vector<Index> topology(
      static_cast<std::size_t>(mesh.numElems()) * static_cast<std::size_t>(cell_nodes));
  for (Index ic = 0; ic < mesh.numElems(); ++ic)
  {
    const auto& cell = mesh.cell(ic);
    if (cell.numNodes() != cell_nodes || cell.shape() != shape)
    {
      throw std::runtime_error("TimeSeriesDataOut supports one cell type per mesh");
    }
    const Index* nodes = mesh.cellNodeIds(ic);
    for (Index i = 0; i < cell_nodes; ++i)
    {
      topology[static_cast<std::size_t>(ic) * static_cast<std::size_t>(cell_nodes) + static_cast<std::size_t>(i)] = nodes[i];
    }
  }

  writeDoubleDataset(file,
                     "/Mesh/Geometry",
                     geometry,
                     {static_cast<hsize_t>(mesh.numNodes()), 3});
  writeIntDataset(file,
                  "/Mesh/Topology",
                  topology,
                  {static_cast<hsize_t>(mesh.numElems()),
                   static_cast<hsize_t>(cell_nodes)});
}

void writeHdf5(const std::string&                          filename,
               const Mesh&                                 mesh,
               const std::vector<TimeSeriesDataOut::Step>& steps)
{
  hid_t file =
      H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  checkHdf5Id(file, "Failed to create HDF5 file: " + filename);

  writeMesh(file, mesh);

  for (std::size_t s = 0; s < steps.size(); ++s)
  {
    const std::string group_path =
        "/Data/" + stepName(static_cast<Index>(s));
    hid_t step_group =
        H5Gcreate2(file,
                   group_path.c_str(),
                   H5P_DEFAULT,
                   H5P_DEFAULT,
                   H5P_DEFAULT);
    checkHdf5Id(step_group, "Failed to create " + group_path);
    checkHdf5(H5Gclose(step_group), "Failed to close " + group_path);

    for (const auto& scalar : steps[s].scalars)
    {
      writeScalarDataset(file, group_path + "/" + scalar.name, scalar.values);
    }
    for (const auto& vec : steps[s].vecs)
    {
      writeVectorDataset(file, group_path + "/" + vec.name, vec.values);
    }
  }

  checkHdf5(H5Fclose(file), "Failed to close HDF5 file: " + filename);
}

#endif

void writeXdmf(const std::string&                          filename,
               const std::string&                          hdf5_filename,
               const Mesh&                                 mesh,
               const std::vector<TimeSeriesDataOut::Step>& steps)
{
  std::ofstream out(filename);
  if (!out)
  {
    throw std::runtime_error("Failed to open XDMF file for writing: " + filename);
  }

  const std::string h5_ref = filenameOnly(hdf5_filename);

  out << R"(<?xml version="1.0" ?>)" << '\n';
  out << R"(<Xdmf Version="3.0">)" << '\n';
  out << "  <Domain>\n";
  out << R"(    <Grid Name="time_series" GridType="Collection" CollectionType="Temporal">)" << '\n';

  for (std::size_t s = 0; s < steps.size(); ++s)
  {
    const std::string step = stepName(static_cast<Index>(s));
    out << R"(      <Grid Name=")" << step << R"(" GridType="Uniform">)" << '\n';
    out << R"(        <Time Value=")" << steps[s].time << R"("/>)" << '\n';
    if (mesh.numElems() == 0)
    {
      throw std::runtime_error("TimeSeriesDataOut needs a non-empty mesh");
    }
    const Index       cell_nodes    = mesh.cells().front().numNodes();
    const Cell::Shape shape         = mesh.cells().front().shape();
    const char*       topology_type = "Triangle";
    if (shape == Cell::Shape::Quadrilateral)
    {
      topology_type = "Quadrilateral";
    }
    else if (shape == Cell::Shape::Tetrahedron)
    {
      topology_type = "Tetrahedron";
    }

    out << R"(        <Topology TopologyType=")" << topology_type
        << R"(" NumberOfElements=")"
        << mesh.numElems() << "\">\n";
    out << R"(          <DataItem Dimensions=")" << mesh.numElems()
        << " " << cell_nodes << R"(" NumberType="Int" Precision="4" Format="HDF">)"
        << h5_ref << ":/Mesh/Topology</DataItem>\n";
    out << "        </Topology>\n";
    out << R"(        <Geometry GeometryType="XYZ">)" << '\n';
    out << R"(          <DataItem Dimensions=")" << mesh.numNodes()
        << R"( 3" NumberType="Float" Precision="8" Format="HDF">)"
        << h5_ref << ":/Mesh/Geometry</DataItem>\n";
    out << "        </Geometry>\n";

    for (const auto& scalar : steps[s].scalars)
    {
      out << R"(        <Attribute Name=")" << scalar.name
          << R"(" AttributeType="Scalar" Center="Node">)" << '\n';
      out << R"(          <DataItem Dimensions=")" << mesh.numNodes()
          << R"(" NumberType="Float" Precision="8" Format="HDF">)"
          << h5_ref << ":/Data/" << step << "/" << scalar.name
          << "</DataItem>\n";
      out << "        </Attribute>\n";
    }

    for (const auto& vec : steps[s].vecs)
    {
      out << R"(        <Attribute Name=")" << vec.name
          << R"(" AttributeType="Vector<Real>" Center="Node">)" << '\n';
      out << R"(          <DataItem Dimensions=")" << mesh.numNodes()
          << R"( 3" NumberType="Float" Precision="8" Format="HDF">)"
          << h5_ref << ":/Data/" << step << "/" << vec.name
          << "</DataItem>\n";
      out << "        </Attribute>\n";
    }

    out << "      </Grid>\n";
  }

  out << "    </Grid>\n";
  out << "  </Domain>\n";
  out << "</Xdmf>\n";
}

} // namespace

void TimeSeriesDataOut::attachMesh(const Mesh& mesh)
{
  mesh_ = &mesh;
}

void TimeSeriesDataOut::beginStep(Real time)
{
  steps_.push_back({});
  steps_.back().time = time;
}

void TimeSeriesDataOut::addNodalScalarField(const std::string&  name,
                                            const Vector<Real>& values)
{
  if (name.empty())
  {
    throw std::runtime_error("TimeSeriesDataOut field name must not be empty");
  }
  currentStep().scalars.push_back({name, values});
}

void TimeSeriesDataOut::addNodalVectorField(const std::string&  name,
                                            const Vector<Real>& x,
                                            const Vector<Real>& y)
{
  Vector<Real> z(x.size());
  z.setZero();
  addNodalVectorField(name, x, y, z);
}

void TimeSeriesDataOut::addNodalVectorField(const std::string&  name,
                                            const Vector<Real>& x,
                                            const Vector<Real>& y,
                                            const Vector<Real>& z)
{
  if (name.empty())
  {
    throw std::runtime_error("TimeSeriesDataOut field name must not be empty");
  }
  currentStep().vecs.push_back({name, {x, y, z}});
}

void TimeSeriesDataOut::clear()
{
  steps_.clear();
}

void TimeSeriesDataOut::write(const std::string& basename) const
{
  checkReady();

  const std::string root          = stripKnownExtension(basename);
  const std::string hdf5_filename = root + ".h5";
  const std::string xdmf_filename = root + ".xdmf";

#ifdef FEMX_HAS_HDF5
  writeHdf5(hdf5_filename, *mesh_, steps_);
#else
  throw std::runtime_error(
      "HDF5 support is not enabled. Configure with FEMX_ENABLE_HDF5=ON "
      "and an available HDF5 C library.");
#endif

  writeXdmf(xdmf_filename, hdf5_filename, *mesh_, steps_);
}

TimeSeriesDataOut::Step& TimeSeriesDataOut::currentStep()
{
  if (steps_.empty())
  {
    throw std::runtime_error(
        "TimeSeriesDataOut::beginStep() must be called before adding fields");
  }
  return steps_.back();
}

void TimeSeriesDataOut::checkReady() const
{
  if (mesh_ == nullptr)
  {
    throw std::runtime_error(
        "TimeSeriesDataOut needs an attached mesh before writing");
  }
  if (steps_.empty())
  {
    throw std::runtime_error("TimeSeriesDataOut has no steps to write");
  }

  checkMesh(*mesh_);

  for (const auto& step : steps_)
  {
    for (const auto& scalar : step.scalars)
    {
      checkFieldSize(*mesh_, scalar.values);
    }
    for (const auto& vec : step.vecs)
    {
      for (const Vector<Real>& component : vec.values)
      {
        checkFieldSize(*mesh_, component);
      }
    }
  }
}

} // namespace femx
