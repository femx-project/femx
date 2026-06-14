#include <stdexcept>
#include <vector>

#include <femx/common/Types.hpp>
#include <femx/io/Hdf5Writer.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Cell.hpp>
#include <femx/mesh/Mesh.hpp>

#ifdef FEMX_HAS_HDF5
#include <hdf5.h>
#endif

namespace femx
{
namespace
{

index_type nodesPerCell(const Mesh& mesh)
{
  if (mesh.numElems() == 0)
  {
    throw std::runtime_error("Hdf5Writer needs a non-empty mesh");
  }

  const index_type nodes = mesh.cells().front().numNodes();
  for (index_type ic = 1; ic < mesh.numElems(); ++ic)
  {
    if (mesh.cell(ic).numNodes() != nodes)
    {
      throw std::runtime_error("Hdf5Writer supports one cell type per mesh");
    }
  }
  return nodes;
}

void checkMeshAndFields(const Mesh&                                mesh,
                        const std::vector<Hdf5Writer::NodalField>& fields)
{
  if (mesh.dim() != 2)
  {
    throw std::runtime_error("Hdf5Writer supports 2D meshes for now");
  }

  const index_type cell_nodes = nodesPerCell(mesh);
  if (cell_nodes != 3 && cell_nodes != 4)
  {
    throw std::runtime_error(
        "Hdf5Writer supports triangle and quadrilateral cells for now");
  }

  for (const auto& field : fields)
  {
    if (field.name.empty())
    {
      throw std::runtime_error("Hdf5Writer field name must not be empty");
    }

    if (field.values == nullptr)
    {
      throw std::runtime_error("Hdf5Writer field has null values");
    }

    if (field.values->size() != mesh.numNodes())
    {
      throw std::runtime_error(
          "Hdf5Writer expects one field value per mesh node");
    }
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
  hid_t dataspace = H5Screate_simple(
      static_cast<int>(dims.size()), dims.data(), nullptr);
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

void writeIntDataset(hid_t                          file,
                     const std::string&             path,
                     const std::vector<index_type>& data,
                     const std::vector<hsize_t>&    dims)
{
  hid_t dataspace = H5Screate_simple(
      static_cast<int>(dims.size()), dims.data(), nullptr);
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

#endif

} // namespace

void Hdf5Writer::write(const std::string&             filename,
                       const Mesh&                    mesh,
                       const std::vector<NodalField>& nodal_fields) const
{
  checkMeshAndFields(mesh, nodal_fields);

#ifdef FEMX_HAS_HDF5
  hid_t file =
      H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  checkHdf5Id(file, "Failed to create HDF5 file: " + filename);

  hid_t mesh_group =
      H5Gcreate2(file, "/Mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  checkHdf5Id(mesh_group, "Failed to create /Mesh group");
  checkHdf5(H5Gclose(mesh_group), "Failed to close /Mesh group");

  hid_t data_group =
      H5Gcreate2(file, "/Data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  checkHdf5Id(data_group, "Failed to create /Data group");
  checkHdf5(H5Gclose(data_group), "Failed to close /Data group");

  std::vector<double> geometry(
      static_cast<std::size_t>(mesh.numNodes()) * 3);
  for (index_type in = 0; in < mesh.numNodes(); ++in)
  {
    for (index_type d = 0; d < 3; ++d)
    {
      geometry[static_cast<std::size_t>(in) * 3 + static_cast<std::size_t>(d)] = mesh.node(in)[d];
    }
  }

  const index_type        nnodes = nodesPerCell(mesh);
  std::vector<index_type> topology(
      static_cast<std::size_t>(mesh.numElems()) * static_cast<std::size_t>(nnodes));
  for (index_type ic = 0; ic < mesh.numElems(); ++ic)
  {
    const index_type* node_ids = mesh.cellNodeIds(ic);
    for (index_type i = 0; i < nnodes; ++i)
    {
      topology[static_cast<std::size_t>(ic) * static_cast<std::size_t>(nnodes) + static_cast<std::size_t>(i)] = node_ids[i];
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
    std::vector<double> values(static_cast<std::size_t>(field.values->size()));
    for (index_type i = 0; i < field.values->size(); ++i)
    {
      values[static_cast<std::size_t>(i)] = (*field.values)[i];
    }

    writeDoubleDataset(file,
                       "/Data/" + field.name,
                       values,
                       {static_cast<hsize_t>(field.values->size())});
  }

  checkHdf5(H5Fclose(file), "Failed to close HDF5 file: " + filename);
#else
  (void) filename;
  throw std::runtime_error(
      "HDF5 support is not enabled. Configure with FEMX_ENABLE_HDF5=ON "
      "and an available HDF5 C library.");
#endif
}

} // namespace femx
