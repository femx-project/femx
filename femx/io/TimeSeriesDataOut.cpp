#include <fstream>
#include <stdexcept>
#include <string>

#include <femx/fem/Element.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>

#ifdef FEMX_HAS_HDF5
#include <hdf5.h>
#endif

namespace femx
{
namespace io
{

using namespace fem;

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
  const size_t pos = path.find_last_of("/\\");
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

  const Index          cn    = mesh.elems().front().numNodes();
  const Element::Shape shape = mesh.elems().front().shape();
  if (shape != Element::Shape::Triangle && shape != Element::Shape::Quadrilateral && shape != Element::Shape::Tetrahedron)
  {
    throw std::runtime_error(
        "TimeSeriesDataOut supports triangle, quadrilateral, and tetrahedron elems");
  }

  for (Index ie = 1; ie < mesh.numElems(); ++ie)
  {
    const auto& elem = mesh.elem(ie);
    if (elem.numNodes() != cn || elem.shape() != shape)
    {
      throw std::runtime_error("TimeSeriesDataOut supports one elem type per mesh");
    }
  }
}

void checkFieldSize(const Mesh& mesh, const HostVector& vals)
{
  if (vals.size() != mesh.numNodes())
  {
    throw std::runtime_error(
        "TimeSeriesDataOut expects one field value per mesh node");
  }
}

#ifdef FEMX_HAS_HDF5

void checkHdf5(herr_t status, const std::string& msg)
{
  if (status < 0)
  {
    throw std::runtime_error(msg);
  }
}

void checkHdf5Id(hid_t id, const std::string& msg)
{
  if (id < 0)
  {
    throw std::runtime_error(msg);
  }
}

void writeDoubleDataset(hid_t                 file,
                        const std::string&    path,
                        const HostVector&     data,
                        const Array<hsize_t>& dims)
{
  hid_t dspace =
      H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
  checkHdf5Id(dspace, "Failed to create HDF5 dataspace for " + path);

  hid_t dset = H5Dcreate2(file,
                          path.c_str(),
                          H5T_NATIVE_DOUBLE,
                          dspace,
                          H5P_DEFAULT,
                          H5P_DEFAULT,
                          H5P_DEFAULT);
  checkHdf5Id(dset, "Failed to create HDF5 dataset " + path);

  checkHdf5(H5Dwrite(dset,
                     H5T_NATIVE_DOUBLE,
                     H5S_ALL,
                     H5S_ALL,
                     H5P_DEFAULT,
                     data.data()),
            "Failed to write HDF5 dataset " + path);

  checkHdf5(H5Dclose(dset), "Failed to close HDF5 dataset " + path);
  checkHdf5(H5Sclose(dspace), "Failed to close HDF5 dataspace " + path);
}

void writeIntDataset(hid_t                 file,
                     const std::string&    path,
                     const Array<Index>&   data,
                     const Array<hsize_t>& dims)
{
  hid_t dspace =
      H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
  checkHdf5Id(dspace, "Failed to create HDF5 dataspace for " + path);

  hid_t dset = H5Dcreate2(file,
                          path.c_str(),
                          H5T_NATIVE_INT,
                          dspace,
                          H5P_DEFAULT,
                          H5P_DEFAULT,
                          H5P_DEFAULT);
  checkHdf5Id(dset, "Failed to create HDF5 dataset " + path);

  checkHdf5(H5Dwrite(dset,
                     H5T_NATIVE_INT,
                     H5S_ALL,
                     H5S_ALL,
                     H5P_DEFAULT,
                     data.data()),
            "Failed to write HDF5 dataset " + path);

  checkHdf5(H5Dclose(dset), "Failed to close HDF5 dataset " + path);
  checkHdf5(H5Sclose(dspace), "Failed to close HDF5 dataspace " + path);
}

void writeScalarDataset(hid_t              file,
                        const std::string& path,
                        const HostVector&  vals)
{
  HostVector data(vals.size());
  for (Index i = 0; i < vals.size(); ++i)
  {
    data[i] = vals[i];
  }

  writeDoubleDataset(file,
                     path,
                     data,
                     {static_cast<hsize_t>(vals.size())});
}

void writeVectorDataset(hid_t                            file,
                        const std::string&               path,
                        const std::array<HostVector, 3>& vals)
{
  const Index num_nodes = vals[0].size();
  HostVector  data(num_nodes * 3);

  for (Index in = 0; in < num_nodes; ++in)
  {
    for (Index d = 0; d < 3; ++d)
    {
      data[in * 3 + d] = vals[d][in];
    }
  }

  writeDoubleDataset(file,
                     path,
                     data,
                     {static_cast<hsize_t>(num_nodes), 3});
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

  HostVector geometry(mesh.numNodes() * 3);
  for (Index in = 0; in < mesh.numNodes(); ++in)
  {
    for (Index d = 0; d < 3; ++d)
    {
      geometry[in * 3 + d] = mesh.node(in)[d];
    }
  }

  if (mesh.numElems() == 0)
  {
    throw std::runtime_error("TimeSeriesDataOut needs a non-empty mesh");
  }

  const Index          cn    = mesh.elems().front().numNodes();
  const Element::Shape shape = mesh.elems().front().shape();
  if (shape != Element::Shape::Triangle && shape != Element::Shape::Quadrilateral && shape != Element::Shape::Tetrahedron)
  {
    throw std::runtime_error(
        "TimeSeriesDataOut supports triangle, quadrilateral, and tetrahedron elems");
  }

  Array<Index> topology(mesh.numElems() * cn);
  for (Index ie = 0; ie < mesh.numElems(); ++ie)
  {
    const auto& elem = mesh.elem(ie);
    if (elem.numNodes() != cn || elem.shape() != shape)
    {
      throw std::runtime_error("TimeSeriesDataOut supports one elem type per mesh");
    }
    const Index* nids = mesh.elemNodeIds(ie);
    for (Index i = 0; i < cn; ++i)
    {
      topology[ie * cn + i] = nids[i];
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
                   static_cast<hsize_t>(cn)});
}

void writeHdf5(const std::string&                    fname,
               const Mesh&                           mesh,
               const Array<TimeSeriesDataOut::Step>& steps)
{
  hid_t file =
      H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  checkHdf5Id(file, "Failed to create HDF5 file: " + fname);

  writeMesh(file, mesh);

  for (Index s = 0; s < steps.size(); ++s)
  {
    const std::string group_path = "/Data/" + stepName(s);
    hid_t             step_group =
        H5Gcreate2(file,
                   group_path.c_str(),
                   H5P_DEFAULT,
                   H5P_DEFAULT,
                   H5P_DEFAULT);
    checkHdf5Id(step_group, "Failed to create " + group_path);
    checkHdf5(H5Gclose(step_group), "Failed to close " + group_path);

    for (const auto& scalar : steps[s].scalars)
    {
      writeScalarDataset(file, group_path + "/" + scalar.name, scalar.vals);
    }
    for (const auto& vec : steps[s].vecs)
    {
      writeVectorDataset(file, group_path + "/" + vec.name, vec.vals);
    }
  }

  checkHdf5(H5Fclose(file), "Failed to close HDF5 file: " + fname);
}

#endif

void writeXdmf(const std::string&                    fname,
               const std::string&                    hdf5_filename,
               const Mesh&                           mesh,
               const Array<TimeSeriesDataOut::Step>& steps)
{
  std::ofstream out(fname);
  if (!out)
  {
    throw std::runtime_error("Failed to open XDMF file for writing: " + fname);
  }

  const std::string h5_ref = filenameOnly(hdf5_filename);

  out << R"(<?xml version="1.0" ?>)" << '\n';
  out << R"(<Xdmf Version="3.0">)" << '\n';
  out << "  <Domain>\n";
  out << R"(    <Grid Name="time_series" GridType="Collection" CollectionType="Temporal">)" << '\n';

  for (Index s = 0; s < steps.size(); ++s)
  {
    const std::string step = stepName(s);
    out << R"(      <Grid Name=")" << step << R"(" GridType="Uniform">)" << '\n';
    out << R"(        <Time Value=")" << steps[s].time << R"("/>)" << '\n';
    if (mesh.numElems() == 0)
    {
      throw std::runtime_error("TimeSeriesDataOut needs a non-empty mesh");
    }
    const Index          cn            = mesh.elems().front().numNodes();
    const Element::Shape shape         = mesh.elems().front().shape();
    const char*          topology_type = "Triangle";
    if (shape == Element::Shape::Quadrilateral)
    {
      topology_type = "Quadrilateral";
    }
    else if (shape == Element::Shape::Tetrahedron)
    {
      topology_type = "Tetrahedron";
    }

    out << R"(        <Topology TopologyType=")" << topology_type
        << R"(" NumberOfElements=")"
        << mesh.numElems() << "\">\n";
    out << R"(          <DataItem Dimensions=")" << mesh.numElems()
        << " " << cn << R"(" NumberType="Int" Precision="4" Format="HDF">)"
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
          << R"(" AttributeType="Vector" Center="Node">)" << '\n';
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

void TimeSeriesDataOut::addNodalScalarField(const std::string& name,
                                            const HostVector&  vals)
{
  if (name.empty())
  {
    throw std::runtime_error("TimeSeriesDataOut field name must not be empty");
  }
  currStep().scalars.push_back({name, vals});
}

void TimeSeriesDataOut::addNodalVectorField(const std::string& name,
                                            const HostVector&  x,
                                            const HostVector&  y)
{
  HostVector z(x.size());
  z.setZero();
  addNodalVectorField(name, x, y, z);
}

void TimeSeriesDataOut::addNodalVectorField(const std::string& name,
                                            const HostVector&  x,
                                            const HostVector&  y,
                                            const HostVector&  z)
{
  if (name.empty())
  {
    throw std::runtime_error("TimeSeriesDataOut field name must not be empty");
  }
  currStep().vecs.push_back({name, {x, y, z}});
}

void TimeSeriesDataOut::clear()
{
  steps_.clear();
}

void TimeSeriesDataOut::write(const std::string& base) const
{
  checkReady();

  const std::string root          = stripKnownExtension(base);
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

TimeSeriesDataOut::Step& TimeSeriesDataOut::currStep()
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
      checkFieldSize(*mesh_, scalar.vals);
    }
    for (const auto& vec : step.vecs)
    {
      for (const HostVector& comp : vec.vals)
      {
        checkFieldSize(*mesh_, comp);
      }
    }
  }
}

} // namespace io
} // namespace femx
