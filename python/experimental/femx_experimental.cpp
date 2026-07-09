#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <femx/fem/GmshReader.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/VelocityProfile.hpp>

namespace py = pybind11;

namespace
{

using femx::Index;
using femx::Mesh;
using femx::Real;

std::string shapeName(femx::Element::Shape shape)
{
  switch (shape)
  {
  case femx::Element::Shape::Unknown:
    return "unknown";
  case femx::Element::Shape::Segment:
    return "segment";
  case femx::Element::Shape::Triangle:
    return "triangle";
  case femx::Element::Shape::Quadrilateral:
    return "quadrilateral";
  case femx::Element::Shape::Tetrahedron:
    return "tetrahedron";
  case femx::Element::Shape::Hexahedron:
    return "hexahedron";
  }
  return "unknown";
}

py::tuple pointTuple(const std::array<Real, 3>& point)
{
  return py::make_tuple(point[0], point[1], point[2]);
}

py::object boundsObject(const Mesh& mesh)
{
  if (mesh.numNodes() == 0)
  {
    return py::none();
  }

  std::array<Real, 3> lower = {
      std::numeric_limits<Real>::max(),
      std::numeric_limits<Real>::max(),
      std::numeric_limits<Real>::max()};
  std::array<Real, 3> upper = {
      std::numeric_limits<Real>::lowest(),
      std::numeric_limits<Real>::lowest(),
      std::numeric_limits<Real>::lowest()};

  for (Index i = 0; i < mesh.numNodes(); ++i)
  {
    const auto& point = mesh.node(i);
    for (Index d = 0; d < 3; ++d)
    {
      lower[d] = std::min(lower[d], point[d]);
      upper[d] = std::max(upper[d], point[d]);
    }
  }

  return py::make_tuple(pointTuple(lower), pointTuple(upper));
}

py::list physicalNamesList(const Mesh& mesh)
{
  py::list names;
  for (const auto& item : mesh.physicalNames())
  {
    py::dict entry;
    entry["dim"]  = item.first.first;
    entry["tag"]  = item.first.second;
    entry["name"] = item.second;
    names.append(entry);
  }
  return names;
}

py::dict elementShapeCounts(const Mesh& mesh)
{
  std::map<std::string, Index> counts;
  for (const auto& elem : mesh.elems())
  {
    ++counts[shapeName(elem.shape())];
  }

  py::dict out;
  for (const auto& item : counts)
  {
    out[py::str(item.first)] = item.second;
  }
  return out;
}

py::list boundaryFacetCounts(const Mesh& mesh)
{
  using Key = std::tuple<Index, Index, std::string, std::string>;
  std::map<Key, Index> counts;
  for (const auto& facet : mesh.boundaryFacets())
  {
    const Key key{
        facet.dim,
        facet.ptag,
        facet.pname,
        shapeName(facet.shape)};
    ++counts[key];
  }

  py::list out;
  for (const auto& item : counts)
  {
    py::dict entry;
    entry["dim"]   = std::get<0>(item.first);
    entry["tag"]   = std::get<1>(item.first);
    entry["name"]  = std::get<2>(item.first);
    entry["shape"] = std::get<3>(item.first);
    entry["count"] = item.second;
    out.append(entry);
  }
  return out;
}

py::dict meshInfo(const Mesh& mesh)
{
  py::dict out;
  out["dim"]                 = mesh.dim();
  out["num_nodes"]           = mesh.numNodes();
  out["num_elements"]        = mesh.numElems();
  out["num_boundary_facets"] = mesh.boundaryFacets().size();
  out["bounds"]              = boundsObject(mesh);
  out["physical_names"]      = physicalNamesList(mesh);
  out["element_shapes"]      = elementShapeCounts(mesh);
  out["boundary_facets"]     = boundaryFacetCounts(mesh);
  return out;
}

py::dict readMeshInfo(const std::string& path)
{
  return meshInfo(femx::GmshReader::read(path));
}

py::tuple boundaryCenter(const std::string& path,
                         const py::object&  boundary)
{
  const Mesh mesh = femx::GmshReader::read(path);

  if (py::isinstance<py::int_>(boundary)
      && !py::isinstance<py::bool_>(boundary))
  {
    return pointTuple(femx::fem::boundaryCenter(
        mesh,
        boundary.cast<Index>()));
  }
  if (py::isinstance<py::str>(boundary))
  {
    return pointTuple(femx::fem::boundaryCenter(
        mesh,
        boundary.cast<std::string>()));
  }

  throw std::runtime_error("boundary must be a physical tag or name");
}

py::dict structuredQuadInfo(Index nx,
                            Index ny,
                            Real  x_min,
                            Real  x_max,
                            Real  y_min,
                            Real  y_max)
{
  if (nx <= 0 || ny <= 0)
  {
    throw std::runtime_error("nx and ny must be positive");
  }
  return meshInfo(Mesh::makeStructuredQuad(nx, ny, x_min, x_max, y_min, y_max));
}

} // namespace

PYBIND11_MODULE(femx_experimental, m)
{
  m.doc() = "Experimental Python bridge for femx mesh utilities.";

  m.def("read_mesh_info",
        &readMeshInfo,
        py::arg("path"),
        "Read a Gmsh .msh file with femx and return a mesh summary.");
  m.def("boundary_center",
        &boundaryCenter,
        py::arg("path"),
        py::arg("boundary"),
        "Return the weighted center of a boundary tag or physical name.");
  m.def("structured_quad_info",
        &structuredQuadInfo,
        py::arg("nx"),
        py::arg("ny"),
        py::arg("x_min") = 0.0,
        py::arg("x_max") = 1.0,
        py::arg("y_min") = 0.0,
        py::arg("y_max") = 1.0,
        "Create a femx structured quad mesh and return a mesh summary.");
}
