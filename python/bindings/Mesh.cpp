#include <memory>
#include <string>

#include "Bindings.hpp"
#include <femx/fem/BoundarySurface.hpp>
#include <femx/fem/GmshReader.hpp>
#include <femx/fem/Mesh.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace
{

using femx::Index;
using femx::Real;
using femx::fem::BoundaryScalarMatrices;
using femx::fem::BoundarySurface;
using femx::fem::Mesh;
using femx::fem::SparseTripletMatrix;

py::array_t<Index> indexArray(const femx::Array<Index>& vals)
{
  py::array_t<Index> out(vals.size());
  auto               data = out.mutable_unchecked<1>();
  for (Index i = 0; i < vals.size(); ++i)
  {
    data(i) = vals[i];
  }
  return out;
}

py::array_t<Real> realArray(const femx::HostVector& vals)
{
  py::array_t<Real> out(vals.size());
  auto              data = out.mutable_unchecked<1>();
  for (Index i = 0; i < vals.size(); ++i)
  {
    data(i) = vals[i];
  }
  return out;
}

py::array_t<Real> meshCoordinates(const Mesh& mesh)
{
  py::array_t<Real> out({mesh.numNodes(), Index{3}});
  auto              data = out.mutable_unchecked<2>();
  for (Index i = 0; i < mesh.numNodes(); ++i)
  {
    for (Index d = 0; d < 3; ++d)
    {
      data(i, d) = mesh.node(i)[d];
    }
  }
  return out;
}

py::array_t<Real> surfaceCoordinates(const BoundarySurface& surface)
{
  py::array_t<Real> out({surface.numNodes(), Index{3}});
  auto              data = out.mutable_unchecked<2>();
  for (Index i = 0; i < surface.numNodes(); ++i)
  {
    for (Index d = 0; d < 3; ++d)
    {
      data(i, d) = surface.nodes()[i][d];
    }
  }
  return out;
}

py::array_t<Index> surfaceElements(const BoundarySurface& surface)
{
  if (surface.numElements() <= 0)
  {
    return py::array_t<Index>(py::array::ShapeContainer{0, 0});
  }
  const Index        nodes_per_element = surface.elements().front().size();
  py::array_t<Index> out({surface.numElements(), nodes_per_element});
  auto               data = out.mutable_unchecked<2>();
  for (Index ie = 0; ie < surface.numElements(); ++ie)
  {
    if (surface.elements()[ie].size() != nodes_per_element)
    {
      throw std::runtime_error("Boundary surface contains mixed element sizes");
    }
    for (Index i = 0; i < nodes_per_element; ++i)
    {
      data(ie, i) = surface.elements()[ie][i];
    }
  }
  return out;
}

py::dict tripletData(const SparseTripletMatrix& mat)
{
  py::dict out;
  out["shape"] = py::make_tuple(mat.rows, mat.cols);
  out["rows"]  = indexArray(mat.row_indices);
  out["cols"]  = indexArray(mat.col_indices);
  out["data"]  = realArray(mat.vals);
  return out;
}

py::dict scalarMatrixData(const BoundarySurface& surface)
{
  const BoundaryScalarMatrices matrices = surface.scalarMatrices();
  py::dict                     out;
  out["stiffness"] = tripletData(matrices.stiffness);
  out["mass"]      = tripletData(matrices.mass);
  out["load"]      = realArray(matrices.load);
  return out;
}

py::array_t<Index> rimMeshNodeIds(const BoundarySurface& surface)
{
  py::array_t<Index> out(surface.rimNodeIds().size());
  auto               data = out.mutable_unchecked<1>();
  for (Index i = 0; i < surface.rimNodeIds().size(); ++i)
  {
    data(i) = surface.meshNodeIds()[surface.rimNodeIds()[i]];
  }
  return out;
}

py::list physicalNames(const Mesh& mesh)
{
  py::list out;
  for (const auto& item : mesh.physicalNames())
  {
    py::dict entry;
    entry["dimension"] = item.first.first;
    entry["tag"]       = item.first.second;
    entry["name"]      = item.second;
    out.append(std::move(entry));
  }
  return out;
}

BoundarySurface selectBoundary(const Mesh& mesh, const py::object& selector)
{
  if (py::isinstance<py::str>(selector))
  {
    return BoundarySurface(mesh, selector.cast<std::string>());
  }
  if (py::isinstance<py::int_>(selector)
      && !py::isinstance<py::bool_>(selector))
  {
    return BoundarySurface(mesh, selector.cast<Index>());
  }
  throw std::runtime_error("Boundary selector must be a physical name or tag");
}

} // namespace

void bindMesh(py::module_& module)
{
  py::class_<BoundarySurface>(module, "BoundarySurface")
      .def_property_readonly("dimension", &BoundarySurface::dim)
      .def_property_readonly("num_nodes", &BoundarySurface::numNodes)
      .def_property_readonly("num_elements", &BoundarySurface::numElements)
      .def_property_readonly(
          "mesh_node_ids",
          [](const BoundarySurface& surface)
          {
            return indexArray(surface.meshNodeIds());
          })
      .def_property_readonly("coordinates", &surfaceCoordinates)
      .def_property_readonly("elements", &surfaceElements)
      .def_property_readonly(
          "rim_node_ids",
          [](const BoundarySurface& surface)
          {
            return indexArray(surface.rimNodeIds());
          })
      .def_property_readonly("rim_mesh_node_ids", &rimMeshNodeIds)
      .def("scalar_matrix_data", &scalarMatrixData);

  py::class_<Mesh, std::shared_ptr<Mesh>>(module, "Mesh")
      .def_static(
          "read",
          [](const std::string& path)
          {
            return std::make_shared<Mesh>(femx::fem::GmshReader::read(path));
          },
          py::arg("path"))
      .def_property_readonly("dimension", &Mesh::dim)
      .def_property_readonly("num_nodes", &Mesh::numNodes)
      .def_property_readonly("num_elements", &Mesh::numElems)
      .def_property_readonly("coordinates", &meshCoordinates)
      .def_property_readonly("physical_names", &physicalNames)
      .def(
          "boundary",
          [](const Mesh& mesh, const py::object& selector)
          {
            return selectBoundary(mesh, selector);
          },
          py::arg("selector"));
}
