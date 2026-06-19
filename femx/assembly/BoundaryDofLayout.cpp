#include <stdexcept>

#include <femx/assembly/BoundaryDofLayout.hpp>

namespace femx
{
namespace assembly
{

BoundaryDofLayout::BoundaryDofLayout(const FESpace& space,
                                     Index          physical_tag)
  : mesh_(&space.mesh()),
    num_dofs_(space.numDofs())
{
  buildByTag(physical_tag,
             [&space](const Mesh::BoundaryFacet& facet, Vector<Index>& dofs)
             { appendFacetDofs(space, facet, dofs); });
}

BoundaryDofLayout::BoundaryDofLayout(const FESpace&     space,
                                     const std::string& physical_name)
  : mesh_(&space.mesh()),
    num_dofs_(space.numDofs())
{
  buildByName(physical_name,
              [&space](const Mesh::BoundaryFacet& facet, Vector<Index>& dofs)
              { appendFacetDofs(space, facet, dofs); });
}

BoundaryDofLayout::BoundaryDofLayout(const MixedFESpace& space,
                                     Index               physical_tag)
  : mesh_(&space.mesh()),
    num_dofs_(space.numDofs())
{
  buildByTag(physical_tag,
             [&space](const Mesh::BoundaryFacet& facet, Vector<Index>& dofs)
             { appendFacetDofs(space, facet, dofs); });
}

BoundaryDofLayout::BoundaryDofLayout(const MixedFESpace& space,
                                     const std::string&  physical_name)
  : mesh_(&space.mesh()),
    num_dofs_(space.numDofs())
{
  buildByName(physical_name,
              [&space](const Mesh::BoundaryFacet& facet, Vector<Index>& dofs)
              { appendFacetDofs(space, facet, dofs); });
}

BoundaryDofLayout BoundaryDofLayout::compact(const FESpace& space,
                                             Index          physical_tag)
{
  return compactByTag(space, physical_tag);
}

BoundaryDofLayout BoundaryDofLayout::compact(
    const FESpace&     space,
    const std::string& physical_name)
{
  return compactByName(space, physical_name);
}

BoundaryDofLayout BoundaryDofLayout::compact(const MixedFESpace& space,
                                             Index               physical_tag)
{
  return compactByTag(space, physical_tag);
}

BoundaryDofLayout BoundaryDofLayout::compact(
    const MixedFESpace& space,
    const std::string&  physical_name)
{
  return compactByName(space, physical_name);
}

Index BoundaryDofLayout::numFacets() const
{
  return facet_dofs_.numSets();
}

Index BoundaryDofLayout::numDofs() const
{
  return num_dofs_;
}

const Mesh::BoundaryFacet& BoundaryDofLayout::facet(Index ib) const
{
  checkFacetIndex(ib);
  const Index mesh_index = facet_indices_[ib];
  return mesh().boundaryFacets()[static_cast<std::size_t>(mesh_index)];
}

Index BoundaryDofLayout::meshFacetIndex(Index ib) const
{
  checkFacetIndex(ib);
  return facet_indices_[ib];
}

void BoundaryDofLayout::facetDofs(Index          ib,
                                  Vector<Index>& dofs) const
{
  checkFacetIndex(ib);
  dofs = facet_dofs_.set(ib);
}

void BoundaryDofLayout::appendFacetDofs(const FESpace&             space,
                                        const Mesh::BoundaryFacet& facet,
                                        Vector<Index>&             dofs)
{
  for (Index node : facet.node_ids)
  {
    for (Index c = 0; c < space.numComponents(); ++c)
    {
      dofs.push_back(space.globalDof(node, c));
    }
  }
}

void BoundaryDofLayout::appendFacetDofs(const MixedFESpace&        space,
                                        const Mesh::BoundaryFacet& facet,
                                        Vector<Index>&             dofs)
{
  for (Index field_id = 0; field_id < space.numFields(); ++field_id)
  {
    const MixedFieldView field = space.field(field_id);
    for (Index node : facet.node_ids)
    {
      for (Index c = 0; c < field.numComponents(); ++c)
      {
        dofs.push_back(field.globalDof(node, c));
      }
    }
  }
}

void BoundaryDofLayout::checkFacetIndex(Index ib) const
{
  if (ib < 0 || ib >= numFacets())
  {
    throw std::runtime_error("BoundaryDofLayout facet index is out of range");
  }
}

const Mesh& BoundaryDofLayout::mesh() const
{
  if (mesh_ == nullptr)
  {
    throw std::runtime_error("BoundaryDofLayout is not initialized");
  }
  return *mesh_;
}

} // namespace assembly
} // namespace femx
