#include <refem/mesh/GmshReader.hpp>

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace refem
{
namespace
{

struct EntityKey
{
  index_type dim = 0;
  index_type tag = 0;

  bool operator<(const EntityKey& other) const noexcept
  {
    return std::tie(dim, tag) < std::tie(other.dim, other.tag);
  }
};

struct EntityData
{
  std::vector<index_type> physical_tags;
};

struct ElementRecord
{
  index_type              entity_dim  = 0;
  index_type              entity_tag  = 0;
  index_type              physical_tag = 0;
  Cell::Shape             shape       = Cell::Shape::Unknown;
  std::vector<index_type> node_ids;
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

Cell::Shape gmshShape(index_type element_type)
{
  switch (element_type)
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

index_type gmshNumNodes(index_type element_type)
{
  switch (element_type)
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
    throw std::runtime_error("GmshReader: unsupported element type " +
                             std::to_string(element_type));
  }
}

index_type gmshElementDimension(index_type element_type)
{
  switch (element_type)
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
    throw std::runtime_error("GmshReader: unsupported element type " +
                             std::to_string(element_type));
  }
}

index_type firstPhysicalTag(const std::map<EntityKey, EntityData>& entities,
                            index_type                             dim,
                            index_type                             tag)
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
  index_type count = 0;
  in >> count;
  for (index_type i = 0; i < count; ++i)
  {
    index_type dim = 0;
    index_type tag = 0;
    in >> dim >> tag;

    std::string rest;
    std::getline(in, rest);
    mesh.addPhysicalName(dim, tag, stripQuotes(rest));
  }
  expectMarker(in, "$EndPhysicalNames");
}

void readEntityBlock(std::istream&                  in,
                     std::map<EntityKey, EntityData>& entities,
                     index_type                     dim,
                     index_type                     count)
{
  for (index_type i = 0; i < count; ++i)
  {
    index_type tag = 0;
    in >> tag;

    const index_type coords = dim == 0 ? 3 : 6;
    for (index_type j = 0; j < coords; ++j)
    {
      real_type ignored = 0.0;
      in >> ignored;
    }

    index_type num_physical_tags = 0;
    in >> num_physical_tags;
    EntityData entity;
    entity.physical_tags.resize(static_cast<std::size_t>(num_physical_tags));
    for (index_type j = 0; j < num_physical_tags; ++j)
    {
      in >> entity.physical_tags[static_cast<std::size_t>(j)];
    }

    index_type num_bounding_entities = 0;
    if (dim > 0)
    {
      in >> num_bounding_entities;
      for (index_type j = 0; j < num_bounding_entities; ++j)
      {
        index_type ignored = 0;
        in >> ignored;
      }
    }

    entities[{dim, tag}] = std::move(entity);
  }
}

void readEntities(std::istream& in, std::map<EntityKey, EntityData>& entities)
{
  index_type num_points  = 0;
  index_type num_curves  = 0;
  index_type num_surfaces = 0;
  index_type num_volumes = 0;
  in >> num_points >> num_curves >> num_surfaces >> num_volumes;

  readEntityBlock(in, entities, 0, num_points);
  readEntityBlock(in, entities, 1, num_curves);
  readEntityBlock(in, entities, 2, num_surfaces);
  readEntityBlock(in, entities, 3, num_volumes);

  expectMarker(in, "$EndEntities");
}

void readNodesV2(std::istream&                    in,
                 Mesh&                            mesh,
                 std::map<index_type, index_type>& node_index_by_tag)
{
  index_type num_nodes = 0;
  in >> num_nodes;

  for (index_type i = 0; i < num_nodes; ++i)
  {
    index_type node_tag = 0;
    Mesh::Node node{};
    in >> node_tag >> node[0] >> node[1] >> node[2];

    const index_type local_id = mesh.numNodes();
    node_index_by_tag[node_tag] = local_id;
    mesh.addNode(node);
  }

  expectMarker(in, "$EndNodes");
}

void readNodesV4(std::istream&                    in,
                 Mesh&                            mesh,
                 std::map<index_type, index_type>& node_index_by_tag)
{
  index_type num_entity_blocks = 0;
  index_type num_nodes         = 0;
  index_type min_node_tag      = 0;
  index_type max_node_tag      = 0;
  in >> num_entity_blocks >> num_nodes >> min_node_tag >> max_node_tag;

  for (index_type block = 0; block < num_entity_blocks; ++block)
  {
    index_type entity_dim         = 0;
    index_type entity_tag         = 0;
    index_type parametric         = 0;
    index_type num_nodes_in_block = 0;
    in >> entity_dim >> entity_tag >> parametric >> num_nodes_in_block;

    std::vector<index_type> node_tags(static_cast<std::size_t>(num_nodes_in_block));
    for (index_type i = 0; i < num_nodes_in_block; ++i)
    {
      in >> node_tags[static_cast<std::size_t>(i)];
    }

    for (index_type i = 0; i < num_nodes_in_block; ++i)
    {
      Mesh::Node node{};
      in >> node[0] >> node[1] >> node[2];
      if (parametric)
      {
        for (index_type j = 0; j < entity_dim; ++j)
        {
          real_type ignored = 0.0;
          in >> ignored;
        }
      }

      const index_type local_id = mesh.numNodes();
      node_index_by_tag[node_tags[static_cast<std::size_t>(i)]] = local_id;
      mesh.addNode(node);
    }
  }

  expectMarker(in, "$EndNodes");
}

void readElementsV2(std::istream&                         in,
                    const std::map<index_type, index_type>& node_index_by_tag,
                    std::vector<ElementRecord>&              elements)
{
  index_type num_elements = 0;
  in >> num_elements;
  elements.reserve(elements.size() + static_cast<std::size_t>(num_elements));

  for (index_type i = 0; i < num_elements; ++i)
  {
    index_type element_tag  = 0;
    index_type element_type = 0;
    index_type num_tags     = 0;
    in >> element_tag >> element_type >> num_tags;

    std::vector<index_type> tags(static_cast<std::size_t>(num_tags));
    for (index_type j = 0; j < num_tags; ++j)
    {
      in >> tags[static_cast<std::size_t>(j)];
    }

    const index_type num_nodes = gmshNumNodes(element_type);
    const Cell::Shape shape = gmshShape(element_type);

    ElementRecord record;
    record.entity_dim   = gmshElementDimension(element_type);
    record.physical_tag = num_tags > 0 ? tags[0] : 0;
    record.entity_tag   = num_tags > 1 ? tags[1] : 0;
    record.shape        = shape;
    record.node_ids.reserve(static_cast<std::size_t>(num_nodes));

    for (index_type j = 0; j < num_nodes; ++j)
    {
      index_type node_tag = 0;
      in >> node_tag;
      const auto node_it = node_index_by_tag.find(node_tag);
      if (node_it == node_index_by_tag.end())
      {
        throw std::runtime_error("GmshReader: element references unknown node " +
                                 std::to_string(node_tag));
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

void readElementsV4(std::istream&                         in,
                    const std::map<index_type, index_type>& node_index_by_tag,
                    std::vector<ElementRecord>&              elements)
{
  index_type num_entity_blocks = 0;
  index_type num_elements      = 0;
  index_type min_element_tag   = 0;
  index_type max_element_tag   = 0;
  in >> num_entity_blocks >> num_elements >> min_element_tag >> max_element_tag;

  elements.reserve(static_cast<std::size_t>(num_elements));
  for (index_type block = 0; block < num_entity_blocks; ++block)
  {
    index_type entity_dim            = 0;
    index_type entity_tag            = 0;
    index_type element_type          = 0;
    index_type num_elements_in_block = 0;
    in >> entity_dim >> entity_tag >> element_type >> num_elements_in_block;

    const index_type num_nodes = gmshNumNodes(element_type);
    const Cell::Shape shape = gmshShape(element_type);

    for (index_type i = 0; i < num_elements_in_block; ++i)
    {
      index_type element_tag = 0;
      in >> element_tag;

      ElementRecord record;
      record.entity_dim = entity_dim;
      record.entity_tag = entity_tag;
      record.physical_tag = 0;
      record.shape      = shape;
      record.node_ids.reserve(static_cast<std::size_t>(num_nodes));

      for (index_type j = 0; j < num_nodes; ++j)
      {
        index_type node_tag = 0;
        in >> node_tag;
        const auto node_it = node_index_by_tag.find(node_tag);
        if (node_it == node_index_by_tag.end())
        {
          throw std::runtime_error("GmshReader: element references unknown node " +
                                   std::to_string(node_tag));
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

index_type meshDimension(const std::vector<ElementRecord>& elements)
{
  index_type dim = 0;
  for (const auto& element : elements)
  {
    switch (element.shape)
    {
    case Cell::Shape::Triangle:
    case Cell::Shape::Quadrilateral:
      dim = std::max<index_type>(dim, 2);
      break;
    case Cell::Shape::Tetrahedron:
    case Cell::Shape::Hexahedron:
      dim = std::max<index_type>(dim, 3);
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

void addElementsToMesh(Mesh&                                   mesh,
                       const std::vector<ElementRecord>&        elements,
                       const std::map<EntityKey, EntityData>& entities)
{
  for (const auto& element : elements)
  {
    const index_type physical_tag = element.physical_tag > 0
                                        ? element.physical_tag
                                        : firstPhysicalTag(entities,
                                                           element.entity_dim,
                                                           element.entity_tag);
    const std::string physical_name =
        mesh.physicalName(element.entity_dim, physical_tag);

    if (element.entity_dim == mesh.dim())
    {
      mesh.addCell(element.node_ids,
                   element.shape,
                   element.entity_dim,
                   element.entity_tag,
                   physical_tag,
                   physical_name);
    }
    else if (element.entity_dim == mesh.dim() - 1)
    {
      Mesh::BoundaryFacet facet;
      facet.dim           = element.entity_dim;
      facet.entity_tag    = element.entity_tag;
      facet.physical_tag  = physical_tag;
      facet.physical_name = physical_name;
      facet.shape         = element.shape;
      facet.node_ids      = element.node_ids;
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

  Mesh mesh;
  std::map<EntityKey, EntityData> entities;
  std::map<index_type, index_type> node_index_by_tag;
  std::vector<ElementRecord> elements;
  real_type version = 0.0;

  std::string marker;
  while (in >> marker)
  {
    if (marker == "$MeshFormat")
    {
      index_type  file_type = 0;
      index_type  data_size = 0;
      in >> version >> file_type >> data_size;
      if (file_type != 0)
      {
        throw std::runtime_error("GmshReader: only ASCII .msh files are supported");
      }
      if (!((version >= 2.0 && version < 3.0) ||
            (version >= 4.0 && version < 5.0)))
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
  for (index_type i = 0; i < mesh.numNodes(); ++i)
  {
    result.addNode(mesh.node(i));
  }
  addElementsToMesh(result, elements, entities);

  return result;
}

} // namespace refem
