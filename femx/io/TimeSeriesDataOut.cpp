#include <fstream>
#include <stdexcept>
#include <string>

#include <femx/fem/Element.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/io/TimeSeriesDataOut.hpp>

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

string filenameOnly(const string& path)
{
  const size_t pos = path.find_last_of("/\\");
  if (pos == string::npos)
  {
    return path;
  }
  return path.substr(pos + 1);
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

void checkMesh(const Mesh& mesh)
{
  if (mesh.dim() != 2 && mesh.dim() != 3)
  {
    throw runtime_error("TimeSeriesDataOut supports 2D and 3D meshes");
  }

  if (mesh.numElems() == 0)
  {
    throw runtime_error("TimeSeriesDataOut needs a non-empty mesh");
  }

  const Index          cn    = mesh.elems().front().numNodes();
  const Element::Shape shape = mesh.elems().front().shape();
  if (shape != Element::Shape::Triangle && shape != Element::Shape::Quadrilateral && shape != Element::Shape::Tetrahedron)
  {
    throw runtime_error(
        "TimeSeriesDataOut supports triangle, quadrilateral, and tetrahedron elems");
  }

  for (Index ie = 1; ie < mesh.numElems(); ++ie)
  {
    const auto& elem = mesh.elem(ie);
    if (elem.numNodes() != cn || elem.shape() != shape)
    {
      throw runtime_error("TimeSeriesDataOut supports one elem type per mesh");
    }
  }
}

void checkFieldSize(const Mesh& mesh, const Vector<Real>& vals)
{
  if (vals.size() != mesh.numNodes())
  {
    throw runtime_error(
        "TimeSeriesDataOut expects one field value per mesh node");
  }
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

void writeDoubleDataset(hid_t                  file,
                        const string&          path,
                        const Vector<double>&  data,
                        const Vector<hsize_t>& dims)
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

void writeIntDataset(hid_t                  file,
                     const string&          path,
                     const Vector<Index>&   data,
                     const Vector<hsize_t>& dims)
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

void writeScalarDataset(hid_t               file,
                        const string&       path,
                        const Vector<Real>& vals)
{
  Vector<double> data(vals.size());
  for (Index i = 0; i < vals.size(); ++i)
  {
    data[i] = vals[i];
  }

  writeDoubleDataset(file,
                     path,
                     data,
                     {static_cast<hsize_t>(vals.size())});
}

void writeVectorDataset(hid_t                         file,
                        const string&                 path,
                        const array<Vector<Real>, 3>& vals)
{
  const Index    nodes = vals[0].size();
  Vector<double> data(nodes * 3);

  for (Index in = 0; in < nodes; ++in)
  {
    for (Index d = 0; d < 3; ++d)
    {
      data[in * 3 + d] = vals[d][in];
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

  Vector<double> geometry(mesh.numNodes() * 3);
  for (Index in = 0; in < mesh.numNodes(); ++in)
  {
    for (Index d = 0; d < 3; ++d)
    {
      geometry[in * 3 + d] = mesh.node(in)[d];
    }
  }

  if (mesh.numElems() == 0)
  {
    throw runtime_error("TimeSeriesDataOut needs a non-empty mesh");
  }

  const Index          cn    = mesh.elems().front().numNodes();
  const Element::Shape shape = mesh.elems().front().shape();
  if (shape != Element::Shape::Triangle && shape != Element::Shape::Quadrilateral && shape != Element::Shape::Tetrahedron)
  {
    throw runtime_error(
        "TimeSeriesDataOut supports triangle, quadrilateral, and tetrahedron elems");
  }

  Vector<Index> topology(mesh.numElems() * cn);
  for (Index ie = 0; ie < mesh.numElems(); ++ie)
  {
    const auto& elem = mesh.elem(ie);
    if (elem.numNodes() != cn || elem.shape() != shape)
    {
      throw runtime_error("TimeSeriesDataOut supports one elem type per mesh");
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

void writeHdf5(const string&                          fname,
               const Mesh&                            mesh,
               const Vector<TimeSeriesDataOut::Step>& steps)
{
  hid_t file =
      H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  checkHdf5Id(file, "Failed to create HDF5 file: " + fname);

  writeMesh(file, mesh);

  for (Index s = 0; s < steps.size(); ++s)
  {
    const string group_path = "/Data/" + stepName(s);
    hid_t        step_group =
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

void writeXdmf(const string&                          fname,
               const string&                          hdf5_filename,
               const Mesh&                            mesh,
               const Vector<TimeSeriesDataOut::Step>& steps)
{
  ofstream out(fname);
  if (!out)
  {
    throw runtime_error("Failed to open XDMF file for writing: " + fname);
  }

  const string h5_ref = filenameOnly(hdf5_filename);

  out << R"(<?xml version="1.0" ?>)" << '\n';
  out << R"(<Xdmf Version="3.0">)" << '\n';
  out << "  <Domain>\n";
  out << R"(    <Grid Name="time_series" GridType="Collection" CollectionType="Temporal">)" << '\n';

  for (Index s = 0; s < steps.size(); ++s)
  {
    const string step = stepName(s);
    out << R"(      <Grid Name=")" << step << R"(" GridType="Uniform">)" << '\n';
    out << R"(        <Time Value=")" << steps[s].time << R"("/>)" << '\n';
    if (mesh.numElems() == 0)
    {
      throw runtime_error("TimeSeriesDataOut needs a non-empty mesh");
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

void TimeSeriesDataOut::addNodalScalarField(const string&       name,
                                            const Vector<Real>& vals)
{
  if (name.empty())
  {
    throw runtime_error("TimeSeriesDataOut field name must not be empty");
  }
  currentStep().scalars.push_back({name, vals});
}

void TimeSeriesDataOut::addNodalVectorField(const string&       name,
                                            const Vector<Real>& x,
                                            const Vector<Real>& y)
{
  Vector<Real> z(x.size());
  z.setZero();
  addNodalVectorField(name, x, y, z);
}

void TimeSeriesDataOut::addNodalVectorField(const string&       name,
                                            const Vector<Real>& x,
                                            const Vector<Real>& y,
                                            const Vector<Real>& z)
{
  if (name.empty())
  {
    throw runtime_error("TimeSeriesDataOut field name must not be empty");
  }
  currentStep().vecs.push_back({name, {x, y, z}});
}

void TimeSeriesDataOut::clear()
{
  steps_.clear();
}

void TimeSeriesDataOut::write(const string& base) const
{
  checkReady();

  const string root          = stripKnownExtension(base);
  const string hdf5_filename = root + ".h5";
  const string xdmf_filename = root + ".xdmf";

#ifdef FEMX_HAS_HDF5
  writeHdf5(hdf5_filename, *mesh_, steps_);
#else
  throw runtime_error(
      "HDF5 support is not enabled. Configure with FEMX_ENABLE_HDF5=ON "
      "and an available HDF5 C library.");
#endif

  writeXdmf(xdmf_filename, hdf5_filename, *mesh_, steps_);
}

TimeSeriesDataOut::Step& TimeSeriesDataOut::currentStep()
{
  if (steps_.empty())
  {
    throw runtime_error(
        "TimeSeriesDataOut::beginStep() must be called before adding fields");
  }
  return steps_.back();
}

void TimeSeriesDataOut::checkReady() const
{
  if (mesh_ == nullptr)
  {
    throw runtime_error(
        "TimeSeriesDataOut needs an attached mesh before writing");
  }
  if (steps_.empty())
  {
    throw runtime_error("TimeSeriesDataOut has no steps to write");
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
      for (const Vector<Real>& comp : vec.vals)
      {
        checkFieldSize(*mesh_, comp);
      }
    }
  }
}

} // namespace femx
