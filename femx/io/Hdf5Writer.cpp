#include <stdexcept>

#include <femx/common/Types.hpp>
#include <femx/fem/Element.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/io/Hdf5Writer.hpp>
#include <femx/linalg/Vector.hpp>

#ifdef FEMX_HAS_HDF5
#include <hdf5.h>
#endif

using namespace std;

namespace femx
{
namespace
{

Index nodesPerElem(const Mesh& mesh)
{
  if (mesh.numElems() == 0)
  {
    throw runtime_error("Hdf5Writer needs a non-empty mesh");
  }

  const Index nodes = mesh.elems().front().numNodes();
  for (Index ie = 1; ie < mesh.numElems(); ++ie)
  {
    if (mesh.elem(ie).numNodes() != nodes)
    {
      throw runtime_error("Hdf5Writer supports one elem type per mesh");
    }
  }
  return nodes;
}

void checkMeshAndFields(const Mesh&                           mesh,
                        const Vector<Hdf5Writer::NodalField>& fields)
{
  if (mesh.dim() != 2)
  {
    throw runtime_error("Hdf5Writer supports 2D meshes for now");
  }

  const Index cn = nodesPerElem(mesh);
  if (cn != 3 && cn != 4)
  {
    throw runtime_error(
        "Hdf5Writer supports triangle and quadrilateral elems for now");
  }

  for (const auto& field : fields)
  {
    if (field.name.empty())
    {
      throw runtime_error("Hdf5Writer field name must not be empty");
    }

    if (field.vals == nullptr)
    {
      throw runtime_error("Hdf5Writer field has null values");
    }

    if (field.vals->size() != mesh.numNodes())
    {
      throw runtime_error(
          "Hdf5Writer expects one field value per mesh node");
    }
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
  hid_t dspace = H5Screate_simple(
      static_cast<int>(dims.size()), dims.data(), nullptr);
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
  hid_t dspace = H5Screate_simple(
      static_cast<int>(dims.size()), dims.data(), nullptr);
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

#endif

} // namespace

void Hdf5Writer::write(const string&             fname,
                       const Mesh&               mesh,
                       const Vector<NodalField>& nodal_fields) const
{
  checkMeshAndFields(mesh, nodal_fields);

#ifdef FEMX_HAS_HDF5
  hid_t file =
      H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  checkHdf5Id(file, "Failed to create HDF5 file: " + fname);

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

  const Index   nnodes = nodesPerElem(mesh);
  Vector<Index> topology(mesh.numElems() * nnodes);
  for (Index ie = 0; ie < mesh.numElems(); ++ie)
  {
    const Index* nids = mesh.elemNodeIds(ie);
    for (Index i = 0; i < nnodes; ++i)
    {
      topology[ie * nnodes + i] = nids[i];
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
                   static_cast<hsize_t>(nnodes)});

  for (const auto& field : nodal_fields)
  {
    Vector<double> vals(field.vals->size());
    for (Index i = 0; i < field.vals->size(); ++i)
    {
      vals[i] = (*field.vals)[i];
    }

    writeDoubleDataset(file,
                       "/Data/" + field.name,
                       vals,
                       {static_cast<hsize_t>(field.vals->size())});
  }

  checkHdf5(H5Fclose(file), "Failed to close HDF5 file: " + fname);
#else
  (void) fname;
  throw runtime_error(
      "HDF5 support is not enabled. Configure with FEMX_ENABLE_HDF5=ON "
      "and an available HDF5 C library.");
#endif
}

} // namespace femx
