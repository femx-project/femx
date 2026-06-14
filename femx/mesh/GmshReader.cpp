#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <femx/mesh/GmshReader.hpp>

namespace femx
{
namespace
{

struct EntityKey
{
  Index dim = 0;
  Index tag = 0;

  bool operator<(const EntityKey& other) const noexcept
  {
    return std::tie(dim, tag) < std::tie(other.dim, other.tag);
  }
};

struct EntityData
{
  std::vector<Index> physical_tags;
};

struct ElementRecord
{
  Index              entity_dim   = 0;
  Index              entity_tag   = 0;
  Index              physical_tag = 0;
  Cell::Shape        shape        = Cell::Shape::Unknown;
  std::vector<Index> node_ids;
};

std::string stripQuotes(std::string value)
{
  const auto first = value.find('"');
  const auto last  = value.rfind('"');
  if (first != std::string::npos && last != std::string::npos && first < last)
  {
    return value.substr(first + 1, last - first - 1);
  }
  return value;
}

void expectMarker(std::istream& in, const std::string& expected)
{
  std::string marker;
  in >> marker;
  if (marker != expected)
  {
    throw std::runtime_error("GmshReader: expected " + expected + ", got " + marker);
  }
}

Cell::Shape gmshShape(Index elem_type)
{
  switch (elem_type)
  {
  case 1:
    return Cell::Shape::Segment;
  case 2:
    return Cell::Shape::Triangle;
  case 3:
    return Cell::Shape::Quadrilateral;
  case 4:
    return Cell::Shape::Tetrahedron;
  case 5:
    return Cell::Shape::Hexahedron;
  default:
    return Cell::Shape::Unknown;
  }
}

Index gmshNumNodes(Index elem_type)
{
  switch (elem_type)
  {
  case 1:
    return 2;
  case 2:
    return 3;
  case 3:
    return 4;
  case 4:
    return 4;
  case 5:
    return 8;
  case 15:
    return 1;
  default:
    throw std::runtime_error("GmshReader: unsupported elem type " + std::to_string(elem_type));
  }
}

Index gmshElementDimension(Index elem_type)
{
  switch (elem_type)
  {
  case 15:
    return 0;
  case 1:
    return 1;
  case 2:
  case 3:
    return 2;
  case 4:
  case 5:
    return 3;
  default:
    throw std::runtime_error("GmshReader: unsupported elem type " + std::to_string(elem_type));
  }
}

Index firstPhysicalTag(const std::map<EntityKey, EntityData>& entities,
                       Index                                  dim,
                       Index                                  tag)
{
  const auto it = entities.find({dim, tag});
  if (it == entities.end() || it->second.physical_tags.empty())
  {
    return 0;
  }
  return it->second.physical_tags.front();
}

void skipUnknownSection(std::istream& in, const std::string& marker)
{
  const std::string end_marker = "$End" + marker.substr(1);
  std::string       token;
  while (in >> token)
  {
    if (token == end_marker)
    {
      return;
    }
  }
  throw std::runtime_error("GmshReader: missing " + end_marker);
}

void readPhysicalNames(std::istream& in, Mesh& mesh)
{
  Index count = 0;
  in >> count;
  for (Index i = 0; i < count; ++i)
  {
    Index dim = 0;
    Index tag = 0;
    in >> dim >> tag;

    std::string rest;
    std::getline(in, rest);
    mesh.addPhysicalName(dim, tag, stripQuotes(rest));
  }
  expectMarker(in, "$EndPhysicalNames");
}

void readEntityBlock(std::istream&                    in,
                     std::map<EntityKey, EntityData>& entities,
                     Index                            dim,
                     Index                            count)
{
  for (Index i = 0; i < count; ++i)
  {
    Index tag = 0;
    in >> tag;

    const Index coords = dim == 0 ? 3 : 6;
    for (Index j = 0; j < coords; ++j)
    {
      Real ignored = 0.0;
      in >> ignored;
    }

    Index num_physical_tags = 0;
    in >> num_physical_tags;
    EntityData entity;
    entity.physical_tags.resize(static_cast<std::size_t>(num_physical_tags));
    for (Index j = 0; j < num_physical_tags; ++j)
    {
      in >> entity.physical_tags[static_cast<std::size_t>(j)];
    }

    Index num_bounding_entities = 0;
    if (dim > 0)
    {
      in >> num_bounding_entities;
      for (Index j = 0; j < num_bounding_entities; ++j)
      {
        Index ignored = 0;
        in >> ignored;
      }
    }

    entities[{dim, tag}] = std::move(entity);
  }
}

void readEntities(std::istream& in, std::map<EntityKey, EntityData>& entities)
{
  Index num_points   = 0;
  Index num_curves   = 0;
  Index num_surfaces = 0;
  Index num_volumes  = 0;
  in >> num_points >> num_curves >> num_surfaces >> num_volumes;

  readEntityBlock(in, entities, 0, num_points);
  readEntityBlock(in, entities, 1, num_curves);
  readEntityBlock(in, entities, 2, num_surfaces);
  readEntityBlock(in, entities, 3, num_volumes);

  expectMarker(in, "$EndEntities");
}

void readNodesV2(std::istream&           in,
                 Mesh&                   mesh,
                 std::map<Index, Index>& node_index_by_tag)
{
  Index num_nodes = 0;
  in >> num_nodes;

  for (Index i = 0; i < num_nodes; ++i)
  {
    Index      node_tag = 0;
    Mesh::Node node{};
    in >> node_tag >> node[0] >> node[1] >> node[2];

    const Index local_id        = mesh.numNodes();
    node_index_by_tag[node_tag] = local_id;
    mesh.addNode(node);
  }

  expectMarker(in, "$EndNodes");
}

void readNodesV4(std::istream&           in,
                 Mesh&                   mesh,
                 std::map<Index, Index>& node_index_by_tag)
{
  Index num_entity_blocks = 0;
  Index num_nodes         = 0;
  Index min_node_tag      = 0;
  Index max_node_tag      = 0;
  in >> num_entity_blocks >> num_nodes >> min_node_tag >> max_node_tag;

  for (Index block = 0; block < num_entity_blocks; ++block)
  {
    Index entity_dim         = 0;
    Index entity_tag         = 0;
    Index parametric         = 0;
    Index num_nodes_in_block = 0;
    in >> entity_dim >> entity_tag >> parametric >> num_nodes_in_block;

    std::vector<Index> node_tags(static_cast<std::size_t>(num_nodes_in_block));
    for (Index i = 0; i < num_nodes_in_block; ++i)
    {
      in >> node_tags[static_cast<std::size_t>(i)];
    }

    for (Index i = 0; i < num_nodes_in_block; ++i)
    {
      Mesh::Node node{};
      in >> node[0] >> node[1] >> node[2];
      if (parametric)
      {
        for (Index j = 0; j < entity_dim; ++j)
        {
          Real ignored = 0.0;
          in >> ignored;
        }
      }

      const Index local_id                                      = mesh.numNodes();
      node_index_by_tag[node_tags[static_cast<std::size_t>(i)]] = local_id;
      mesh.addNode(node);
    }
  }

  expectMarker(in, "$EndNodes");
}

void readElementsV2(std::istream&                 in,
                    const std::map<Index, Index>& node_index_by_tag,
                    std::vector<ElementRecord>&   elements)
{
  Index num_elems = 0;
  in >> num_elems;
  elements.reserve(elements.size() + static_cast<std::size_t>(num_elems));

  for (Index i = 0; i < num_elems; ++i)
  {
    Index elem_tag  = 0;
    Index elem_type = 0;
    Index num_tags     = 0;
    in >> elem_tag >> elem_type >> num_tags;

    std::vector<Index> tags(static_cast<std::size_t>(num_tags));
    for (Index j = 0; j < num_tags; ++j)
    {
      in >> tags[static_cast<std::size_t>(j)];
    }

    const Index       num_nodes = gmshNumNodes(elem_type);
    const Cell::Shape shape     = gmshShape(elem_type);

    ElementRecord record;
    record.entity_dim   = gmshElementDimension(elem_type);
    record.physical_tag = num_tags > 0 ? tags[0] : 0;
    record.entity_tag   = num_tags > 1 ? tags[1] : 0;
    record.shape        = shape;
    record.node_ids.reserve(static_cast<std::size_t>(num_nodes));

    for (Index j = 0; j < num_nodes; ++j)
    {
      Index node_tag = 0;
      in >> node_tag;
      const auto node_it = node_index_by_tag.find(node_tag);
      if (node_it == node_index_by_tag.end())
      {
        throw std::runtime_error("GmshReader: elem references unknown node " + std::to_string(node_tag));
      }
      record.node_ids.push_back(node_it->second);
    }

    if (shape != Cell::Shape::Unknown && record.entity_dim > 0)
    {
      elements.push_back(std::move(record));
    }
  }

  expectMarker(in, "$EndElements");
}

void readElementsV4(std::istream&                 in,
                    const std::map<Index, Index>& node_index_by_tag,
                    std::vector<ElementRecord>&   elements)
{
  Index num_entity_blocks = 0;
  Index num_elems      = 0;
  Index min_elem_tag   = 0;
  Index max_elem_tag   = 0;
  in >> num_entity_blocks >> num_elems >> min_elem_tag >> max_elem_tag;

  elements.reserve(static_cast<std::size_t>(num_elems));
  for (Index block = 0; block < num_entity_blocks; ++block)
  {
    Index entity_dim            = 0;
    Index entity_tag            = 0;
    Index elem_type          = 0;
    Index num_elems_in_block = 0;
    in >> entity_dim >> entity_tag >> elem_type >> num_elems_in_block;

    const Index       num_nodes = gmshNumNodes(elem_type);
    const Cell::Shape shape     = gmshShape(elem_type);

    for (Index i = 0; i < num_elems_in_block; ++i)
    {
      Index elem_tag = 0;
      in >> elem_tag;

      ElementRecord record;
      record.entity_dim   = entity_dim;
      record.entity_tag   = entity_tag;
      record.physical_tag = 0;
      record.shape        = shape;
      record.node_ids.reserve(static_cast<std::size_t>(num_nodes));

      for (Index j = 0; j < num_nodes; ++j)
      {
        Index node_tag = 0;
        in >> node_tag;
        const auto node_it = node_index_by_tag.find(node_tag);
        if (node_it == node_index_by_tag.end())
        {
          throw std::runtime_error("GmshReader: elem references unknown node " + std::to_string(node_tag));
        }
        record.node_ids.push_back(node_it->second);
      }

      if (shape != Cell::Shape::Unknown)
      {
        elements.push_back(std::move(record));
      }
    }
  }

  expectMarker(in, "$EndElements");
}

Index meshDimension(const std::vector<ElementRecord>& elements)
{
  Index dim = 0;
  for (const auto& elem : elements)
  {
    switch (elem.shape)
    {
    case Cell::Shape::Triangle:
    case Cell::Shape::Quadrilateral:
      dim = std::max<Index>(dim, 2);
      break;
    case Cell::Shape::Tetrahedron:
    case Cell::Shape::Hexahedron:
      dim = std::max<Index>(dim, 3);
      break;
    default:
      break;
    }
  }
  if (dim == 0)
  {
    throw std::runtime_error("GmshReader: no supported volume or surface cells found");
  }
  return dim;
}

void addElementsToMesh(Mesh&                                  mesh,
                       const std::vector<ElementRecord>&      elements,
                       const std::map<EntityKey, EntityData>& entities)
{
  for (const auto& elem : elements)
  {
    const Index       physical_tag = elem.physical_tag > 0
                                         ? elem.physical_tag
                                         : firstPhysicalTag(entities,
                                                      elem.entity_dim,
                                                      elem.entity_tag);
    const std::string physical_name =
        mesh.physicalName(elem.entity_dim, physical_tag);

    if (elem.entity_dim == mesh.dim())
    {
      mesh.addCell(elem.node_ids,
                   elem.shape,
                   elem.entity_dim,
                   elem.entity_tag,
                   physical_tag,
                   physical_name);
    }
    else if (elem.entity_dim == mesh.dim() - 1)
    {
      Mesh::BoundaryFacet facet;
      facet.dim           = elem.entity_dim;
      facet.entity_tag    = elem.entity_tag;
      facet.physical_tag  = physical_tag;
      facet.physical_name = physical_name;
      facet.shape         = elem.shape;
      facet.node_ids      = elem.node_ids;
      mesh.addBoundaryFacet(std::move(facet));
    }
  }
}

} // namespace

Mesh GmshReader::read(const std::string& path)
{
  std::ifstream in(path);
  if (!in)
  {
    throw std::runtime_error("GmshReader: failed to open " + path);
  }

  Mesh                            mesh;
  std::map<EntityKey, EntityData> entities;
  std::map<Index, Index>          node_index_by_tag;
  std::vector<ElementRecord>      elements;
  Real                            version = 0.0;

  std::string marker;
  while (in >> marker)
  {
    if (marker == "$MeshFormat")
    {
      Index file_type = 0;
      Index data_size = 0;
      in >> version >> file_type >> data_size;
      if (file_type != 0)
      {
        throw std::runtime_error("GmshReader: only ASCII .msh files are supported");
      }
      if (!((version >= 2.0 && version < 3.0) || (version >= 4.0 && version < 5.0)))
      {
        throw std::runtime_error("GmshReader: only Gmsh 2.x and 4.x .msh files are supported");
      }
      expectMarker(in, "$EndMeshFormat");
    }
    else if (marker == "$PhysicalNames")
    {
      readPhysicalNames(in, mesh);
    }
    else if (marker == "$Entities")
    {
      readEntities(in, entities);
    }
    else if (marker == "$Nodes")
    {
      if (version >= 4.0)
      {
        readNodesV4(in, mesh, node_index_by_tag);
      }
      else
      {
        readNodesV2(in, mesh, node_index_by_tag);
      }
    }
    else if (marker == "$Elements")
    {
      if (version >= 4.0)
      {
        readElementsV4(in, node_index_by_tag, elements);
      }
      else
      {
        readElementsV2(in, node_index_by_tag, elements);
      }
    }
    else
    {
      skipUnknownSection(in, marker);
    }
  }

  Mesh result(meshDimension(elements));
  for (const auto& physical_name : mesh.physicalNames())
  {
    result.addPhysicalName(physical_name.first.first,
                           physical_name.first.second,
                           physical_name.second);
  }
  for (Index i = 0; i < mesh.numNodes(); ++i)
  {
    result.addNode(mesh.node(i));
  }
  addElementsToMesh(result, elements, entities);

  return result;
}

} // namespace femx
