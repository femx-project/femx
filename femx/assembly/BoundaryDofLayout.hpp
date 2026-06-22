#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include <femx/common/Types.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/Mesh.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/IndexSetList.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace assembly
{

/** @brief Boundary-facet-to-global-dof connectivity for node-based spaces. */
class BoundaryDofLayout
{
public:
  BoundaryDofLayout(const FESpace& space,
                    Index          ptag);

  BoundaryDofLayout(const FESpace&     space,
                    const std::string& pname);

  BoundaryDofLayout(const MixedFESpace& space,
                    Index               ptag);

  BoundaryDofLayout(const MixedFESpace& space,
                    const std::string&  pname);

  static BoundaryDofLayout compact(const FESpace& space,
                                   Index          ptag);

  static BoundaryDofLayout compact(const FESpace&     space,
                                   const std::string& pname);

  static BoundaryDofLayout compact(const MixedFESpace& space,
                                   Index               ptag);

  static BoundaryDofLayout compact(const MixedFESpace& space,
                                   const std::string&  pname);

  Index numFacets() const;

  Index numDofs() const;

  const Mesh::BoundaryFacet& facet(Index ib) const;

  Index meshFacetIndex(Index ib) const;

  void facetDofs(Index ib, Vector<Index>& dofs) const;

private:
  BoundaryDofLayout() = default;

  template <typename Space>
  static BoundaryDofLayout compactByTag(const Space& space,
                                        Index        ptag)
  {
    BoundaryDofLayout lyt;
    lyt.mesh_ = &space.mesh();

    std::map<Index, Index> compact_dofs;
    lyt.buildByTag(ptag,
                   [&space, &compact_dofs](
                       const Mesh::BoundaryFacet& facet,
                       Vector<Index>&             dofs)
                   {
                     appendCompactFacetDofs(
                         space, facet, compact_dofs, dofs);
                   });
    lyt.nd_ = static_cast<Index>(compact_dofs.size());
    return lyt;
  }

  template <typename Space>
  static BoundaryDofLayout compactByName(const Space&       space,
                                         const std::string& pname)
  {
    BoundaryDofLayout lyt;
    lyt.mesh_ = &space.mesh();

    std::map<Index, Index> compact_dofs;
    lyt.buildByName(pname,
                    [&space, &compact_dofs](
                        const Mesh::BoundaryFacet& facet,
                        Vector<Index>&             dofs)
                    {
                      appendCompactFacetDofs(
                          space, facet, compact_dofs, dofs);
                    });
    lyt.nd_ = static_cast<Index>(compact_dofs.size());
    return lyt;
  }

  template <typename DofAppender>
  void buildByTag(Index ptag, DofAppender append_dofs)
  {
    const auto& facets = mesh().boundaryFacets();
    for (Index i = 0; i < facets.size(); ++i)
    {
      if (facets[i].ptag == ptag)
      {
        addFacet(i, facets[i], append_dofs);
      }
    }
    if (facet_dofs_.empty())
    {
      throw std::runtime_error(
          "No boundary facets found for physical tag "
          + std::to_string(ptag));
    }
  }

  template <typename DofAppender>
  void buildByName(const std::string& pname, DofAppender append_dofs)
  {
    const auto& facets = mesh().boundaryFacets();
    for (Index i = 0; i < facets.size(); ++i)
    {
      if (facets[i].pname == pname)
      {
        addFacet(i, facets[i], append_dofs);
      }
    }
    if (facet_dofs_.empty())
    {
      throw std::runtime_error(
          "No boundary facets found for physical name " + pname);
    }
  }

  template <typename DofAppender>
  void addFacet(Index                      mesh_facet_index,
                const Mesh::BoundaryFacet& facet,
                DofAppender                append_dofs)
  {
    Vector<Index> dofs;
    append_dofs(facet, dofs);
    if (dofs.empty())
    {
      throw std::runtime_error("BoundaryDofLayout facet has no dofs");
    }
    facet_indices_.push_back(mesh_facet_index);
    facet_dofs_.pushBack(dofs);
  }

  static void appendFacetDofs(const FESpace&             space,
                              const Mesh::BoundaryFacet& facet,
                              Vector<Index>&             dofs);

  static void appendFacetDofs(const MixedFESpace&        space,
                              const Mesh::BoundaryFacet& facet,
                              Vector<Index>&             dofs);

  template <typename Space>
  static void appendCompactFacetDofs(
      const Space&               space,
      const Mesh::BoundaryFacet& facet,
      std::map<Index, Index>&    compact_dofs,
      Vector<Index>&             dofs)
  {
    Vector<Index> full_dofs;
    appendFacetDofs(space, facet, full_dofs);
    dofs.reserve(full_dofs.size());

    for (Index full_dof : full_dofs)
    {
      auto it = compact_dofs.find(full_dof);
      if (it == compact_dofs.end())
      {
        const Index compact_dof =
            static_cast<Index>(compact_dofs.size());
        it = compact_dofs.emplace(full_dof, compact_dof).first;
      }
      dofs.push_back(it->second);
    }
  }

  void checkFacetIndex(Index ib) const;

  const Mesh& mesh() const;

private:
  const Mesh*   mesh_{nullptr};
  Index         nd_{0};
  Vector<Index> facet_indices_;
  IndexSetList  facet_dofs_;
};

} // namespace assembly
} // namespace femx
