#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <femx/fem/GmshReader.hpp>

using namespace std;

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
    return tie(dim, tag) < tie(other.dim, other.tag);
  }
};

struct EntityData
{
  Vector<Index> ptags;
};

struct ElemRecord
{
  Index          edim  = 0;
  Index          etag  = 0;
  Index          ptag  = 0;
  Element::Shape shape = Element::Shape::Unknown;
  Vector<Index>  nids;
};

string stripQuotes(string value)
{
  const auto first = value.find('"');
  const auto last  = value.rfind('"');
  if (first != string::npos && last != string::npos && first < last)
  {
    return value.substr(first + 1, last - first - 1);
  }
  return value;
}

void expectMarker(istream& in, const string& exp)
{
  string mark;
  in >> mark;
  if (mark != exp)
  {
    throw runtime_error("GmshReader: expected " + exp + ", got " + mark);
  }
}

Element::Shape gmshShape(Index elem_type)
{
  switch (elem_type)
  {
  case 1:
    return Element::Shape::Segment;
  case 2:
    return Element::Shape::Triangle;
  case 3:
    return Element::Shape::Quadrilateral;
  case 4:
    return Element::Shape::Tetrahedron;
  case 5:
    return Element::Shape::Hexahedron;
  default:
    return Element::Shape::Unknown;
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
    throw runtime_error("GmshReader: unsupported elem type " + to_string(elem_type));
  }
}

Index gmshElemDim(Index elem_type)
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
    throw runtime_error("GmshReader: unsupported elem type " + to_string(elem_type));
  }
}

Index firstPhysicalTag(const map<EntityKey, EntityData>& entities,
                       Index                             dim,
                       Index                             tag)
{
  const auto it = entities.find({dim, tag});
  if (it == entities.end() || it->second.ptags.empty())
  {
    return 0;
  }
  return it->second.ptags.front();
}

void skipUnknownSection(istream& in, const string& mark)
{
  const string end_marker = "$End" + mark.substr(1);
  string       token;
  while (in >> token)
  {
    if (token == end_marker)
    {
      return;
    }
  }
  throw runtime_error("GmshReader: missing " + end_marker);
}

void readPhysicalNames(istream& in, Mesh& mesh)
{
  Index count = 0;
  in >> count;
  for (Index i = 0; i < count; ++i)
  {
    Index dim = 0;
    Index tag = 0;
    in >> dim >> tag;

    string rest;
    getline(in, rest);
    mesh.addPhysicalName(dim, tag, stripQuotes(rest));
  }
  expectMarker(in, "$EndPhysicalNames");
}

void readEntityBlock(istream&                    in,
                     map<EntityKey, EntityData>& entities,
                     Index                       dim,
                     Index                       count)
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

    Index num_ptags = 0;
    in >> num_ptags;
    EntityData entity;
    entity.ptags.resize(num_ptags);
    for (Index j = 0; j < num_ptags; ++j)
    {
      in >> entity.ptags[j];
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

void readEntities(istream& in, map<EntityKey, EntityData>& entities)
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

void readNodesV2(istream&           in,
                 Mesh&              mesh,
                 map<Index, Index>& nid_by_tag)
{
  Index num_nodes = 0;
  in >> num_nodes;

  for (Index i = 0; i < num_nodes; ++i)
  {
    Index      ntag = 0;
    Mesh::Node node{};
    in >> ntag >> node[0] >> node[1] >> node[2];

    const Index local_id = mesh.numNodes();
    nid_by_tag[ntag]     = local_id;
    mesh.addNode(node);
  }

  expectMarker(in, "$EndNodes");
}

void readNodesV4(istream&           in,
                 Mesh&              mesh,
                 map<Index, Index>& nid_by_tag)
{
  Index num_blocks = 0;
  Index num_nodes         = 0;
  Index min_ntag   = 0;
  Index max_ntag   = 0;
  in >> num_blocks >> num_nodes >> min_ntag >> max_ntag;

  for (Index block = 0; block < num_blocks; ++block)
  {
    Index edim       = 0;
    Index etag       = 0;
    Index parametric = 0;
    Index num_nodes_block   = 0;
    in >> edim >> etag >> parametric >> num_nodes_block;

    Vector<Index> ntags(num_nodes_block);
    for (Index i = 0; i < num_nodes_block; ++i)
    {
      in >> ntags[i];
    }

    for (Index i = 0; i < num_nodes_block; ++i)
    {
      Mesh::Node node{};
      in >> node[0] >> node[1] >> node[2];
      if (parametric)
      {
        for (Index j = 0; j < edim; ++j)
        {
          Real ignored = 0.0;
          in >> ignored;
        }
      }

      const Index local_id = mesh.numNodes();
      nid_by_tag[ntags[i]] = local_id;
      mesh.addNode(node);
    }
  }

  expectMarker(in, "$EndNodes");
}

void readElemsV2(istream&                 in,
                 const map<Index, Index>& nid_by_tag,
                 Vector<ElemRecord>&      elems)
{
  Index num_elems = 0;
  in >> num_elems;
  elems.reserve(elems.size() + num_elems);

  for (Index i = 0; i < num_elems; ++i)
  {
    Index elem_tag  = 0;
    Index elem_type = 0;
    Index num_tags  = 0;
    in >> elem_tag >> elem_type >> num_tags;

    Vector<Index> tags(num_tags);
    for (Index j = 0; j < num_tags; ++j)
    {
      in >> tags[j];
    }

    const Index          num_nodes    = gmshNumNodes(elem_type);
    const Element::Shape shape = gmshShape(elem_type);

    ElemRecord rec;
    rec.edim  = gmshElemDim(elem_type);
    rec.ptag  = num_tags > 0 ? tags[0] : 0;
    rec.etag  = num_tags > 1 ? tags[1] : 0;
    rec.shape = shape;
    rec.nids.reserve(num_nodes);

    for (Index j = 0; j < num_nodes; ++j)
    {
      Index ntag = 0;
      in >> ntag;
      const auto node_it = nid_by_tag.find(ntag);
      if (node_it == nid_by_tag.end())
      {
        throw runtime_error("GmshReader: elem references unknown node " + to_string(ntag));
      }
      rec.nids.push_back(node_it->second);
    }

    if (shape != Element::Shape::Unknown && rec.edim > 0)
    {
      elems.push_back(std::move(rec));
    }
  }

  expectMarker(in, "$EndElements");
}

void readElemsV4(istream&                 in,
                 const map<Index, Index>& nid_by_tag,
                 Vector<ElemRecord>&      elems)
{
  Index num_blocks   = 0;
  Index num_elems           = 0;
  Index min_elem_tag = 0;
  Index max_elem_tag = 0;
  in >> num_blocks >> num_elems >> min_elem_tag >> max_elem_tag;

  elems.reserve(num_elems);
  for (Index block = 0; block < num_blocks; ++block)
  {
    Index edim      = 0;
    Index etag      = 0;
    Index elem_type = 0;
    Index num_elems_block  = 0;
    in >> edim >> etag >> elem_type >> num_elems_block;

    const Index          num_nodes    = gmshNumNodes(elem_type);
    const Element::Shape shape = gmshShape(elem_type);

    for (Index i = 0; i < num_elems_block; ++i)
    {
      Index elem_tag = 0;
      in >> elem_tag;

      ElemRecord rec;
      rec.edim  = edim;
      rec.etag  = etag;
      rec.ptag  = 0;
      rec.shape = shape;
      rec.nids.reserve(num_nodes);

      for (Index j = 0; j < num_nodes; ++j)
      {
        Index ntag = 0;
        in >> ntag;
        const auto node_it = nid_by_tag.find(ntag);
        if (node_it == nid_by_tag.end())
        {
          throw runtime_error("GmshReader: elem references unknown node " + to_string(ntag));
        }
        rec.nids.push_back(node_it->second);
      }

      if (shape != Element::Shape::Unknown)
      {
        elems.push_back(std::move(rec));
      }
    }
  }

  expectMarker(in, "$EndElements");
}

Index meshDim(const Vector<ElemRecord>& elems)
{
  Index dim = 0;
  for (const auto& elem : elems)
  {
    switch (elem.shape)
    {
    case Element::Shape::Triangle:
    case Element::Shape::Quadrilateral:
      dim = max<Index>(dim, 2);
      break;
    case Element::Shape::Tetrahedron:
    case Element::Shape::Hexahedron:
      dim = max<Index>(dim, 3);
      break;
    default:
      break;
    }
  }
  if (dim == 0)
  {
    throw runtime_error("GmshReader: no supported volume or surface elems found");
  }
  return dim;
}

void addElemsToMesh(Mesh&                             mesh,
                    const Vector<ElemRecord>&         elems,
                    const map<EntityKey, EntityData>& entities)
{
  for (const auto& elem : elems)
  {
    const Index  ptag = elem.ptag > 0
                            ? elem.ptag
                            : firstPhysicalTag(entities,
                                              elem.edim,
                                              elem.etag);
    const string pname =
        mesh.physicalName(elem.edim, ptag);

    if (elem.edim == mesh.dim())
    {
      mesh.addElem(elem.nids,
                   elem.shape,
                   elem.edim,
                   elem.etag,
                   ptag,
                   pname);
    }
    else if (elem.edim == mesh.dim() - 1)
    {
      Mesh::BoundaryFacet facet;
      facet.dim   = elem.edim;
      facet.etag  = elem.etag;
      facet.ptag  = ptag;
      facet.pname = pname;
      facet.shape = elem.shape;
      facet.nids  = elem.nids;
      mesh.addBoundaryFacet(std::move(facet));
    }
  }
}

} // namespace

Mesh GmshReader::read(const string& path)
{
  ifstream in(path);
  if (!in)
  {
    throw runtime_error("GmshReader: failed to open " + path);
  }

  Mesh                       mesh;
  map<EntityKey, EntityData> entities;
  map<Index, Index>          nid_by_tag;
  Vector<ElemRecord>         elems;
  Real                       version = 0.0;

  string mark;
  while (in >> mark)
  {
    if (mark == "$MeshFormat")
    {
      Index file_type = 0;
      Index data_size = 0;
      in >> version >> file_type >> data_size;
      if (file_type != 0)
      {
        throw runtime_error("GmshReader: only ASCII .msh files are supported");
      }
      if (!((version >= 2.0 && version < 3.0) || (version >= 4.0 && version < 5.0)))
      {
        throw runtime_error("GmshReader: only Gmsh 2.x and 4.x .msh files are supported");
      }
      expectMarker(in, "$EndMeshFormat");
    }
    else if (mark == "$PhysicalNames")
    {
      readPhysicalNames(in, mesh);
    }
    else if (mark == "$Entities")
    {
      readEntities(in, entities);
    }
    else if (mark == "$Nodes")
    {
      if (version >= 4.0)
      {
        readNodesV4(in, mesh, nid_by_tag);
      }
      else
      {
        readNodesV2(in, mesh, nid_by_tag);
      }
    }
    else if (mark == "$Elements")
    {
      if (version >= 4.0)
      {
        readElemsV4(in, nid_by_tag, elems);
      }
      else
      {
        readElemsV2(in, nid_by_tag, elems);
      }
    }
    else
    {
      skipUnknownSection(in, mark);
    }
  }

  Mesh result(meshDim(elems));
  for (const auto& pname : mesh.physicalNames())
  {
    result.addPhysicalName(pname.first.first,
                           pname.first.second,
                           pname.second);
  }
  for (Index i = 0; i < mesh.numNodes(); ++i)
  {
    result.addNode(mesh.node(i));
  }
  addElemsToMesh(result, elems, entities);

  return result;
}

} // namespace femx
