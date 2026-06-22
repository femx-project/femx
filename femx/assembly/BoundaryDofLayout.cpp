#include <stdexcept>

#include <femx/assembly/BoundaryDofLayout.hpp>

using namespace std;

namespace femx
{
namespace assembly
{

BoundaryDofLayout::BoundaryDofLayout(const FESpace& space,
                                     Index          ptag)
  : mesh_(&space.mesh()),
    nd_(space.numDofs())
{
  buildByTag(ptag,
             [&space](const Mesh::BoundaryFacet& facet, Vector<Index>& dofs)
             { appendFacetDofs(space, facet, dofs); });
}

BoundaryDofLayout::BoundaryDofLayout(const FESpace& space,
                                     const string&  pname)
  : mesh_(&space.mesh()),
    nd_(space.numDofs())
{
  buildByName(pname,
              [&space](const Mesh::BoundaryFacet& facet, Vector<Index>& dofs)
              { appendFacetDofs(space, facet, dofs); });
}

BoundaryDofLayout::BoundaryDofLayout(const MixedFESpace& space,
                                     Index               ptag)
  : mesh_(&space.mesh()),
    nd_(space.numDofs())
{
  buildByTag(ptag,
             [&space](const Mesh::BoundaryFacet& facet, Vector<Index>& dofs)
             { appendFacetDofs(space, facet, dofs); });
}

BoundaryDofLayout::BoundaryDofLayout(const MixedFESpace& space,
                                     const string&       pname)
  : mesh_(&space.mesh()),
    nd_(space.numDofs())
{
  buildByName(pname,
              [&space](const Mesh::BoundaryFacet& facet, Vector<Index>& dofs)
              { appendFacetDofs(space, facet, dofs); });
}

BoundaryDofLayout BoundaryDofLayout::compact(const FESpace& space,
                                             Index          ptag)
{
  return compactByTag(space, ptag);
}

BoundaryDofLayout BoundaryDofLayout::compact(
    const FESpace& space,
    const string&  pname)
{
  return compactByName(space, pname);
}

BoundaryDofLayout BoundaryDofLayout::compact(const MixedFESpace& space,
                                             Index               ptag)
{
  return compactByTag(space, ptag);
}

BoundaryDofLayout BoundaryDofLayout::compact(
    const MixedFESpace& space,
    const string&       pname)
{
  return compactByName(space, pname);
}

Index BoundaryDofLayout::numFacets() const
{
  return facet_dofs_.numSets();
}

Index BoundaryDofLayout::numDofs() const
{
  return nd_;
}

const Mesh::BoundaryFacet& BoundaryDofLayout::facet(Index ib) const
{
  checkFacetIndex(ib);
  const Index mesh_index = facet_indices_[ib];
  return mesh().boundaryFacets()[mesh_index];
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
  for (Index node : facet.nids)
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
  for (Index fid = 0; fid < space.numFields(); ++fid)
  {
    const MixedFieldView field = space.field(fid);
    for (Index node : facet.nids)
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
    throw runtime_error("BoundaryDofLayout facet index is out of range");
  }
}

const Mesh& BoundaryDofLayout::mesh() const
{
  if (mesh_ == nullptr)
  {
    throw runtime_error("BoundaryDofLayout is not initialized");
  }
  return *mesh_;
}

} // namespace assembly
} // namespace femx
