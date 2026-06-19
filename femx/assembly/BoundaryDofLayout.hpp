#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <femx/common/Types.hpp>
#include <femx/fem/FESpace.hpp>
#include <femx/fem/MixedFESpace.hpp>
#include <femx/linalg/IndexSetList.hpp>
#include <femx/linalg/Vector.hpp>
#include <femx/mesh/Mesh.hpp>

namespace femx
{
namespace assembly
{

/** @brief Boundary-facet-to-global-dof connectivity for node-based spaces. */
class BoundaryDofLayout
{
public:
  BoundaryDofLayout(const FESpace& space,
                    Index          physical_tag);

  BoundaryDofLayout(const FESpace&     space,
                    const std::string& physical_name);

  BoundaryDofLayout(const MixedFESpace& space,
                    Index               physical_tag);

  BoundaryDofLayout(const MixedFESpace& space,
                    const std::string&  physical_name);

  static BoundaryDofLayout compact(const FESpace& space,
                                   Index          physical_tag);

  static BoundaryDofLayout compact(const FESpace&     space,
                                   const std::string& physical_name);

  static BoundaryDofLayout compact(const MixedFESpace& space,
                                   Index               physical_tag);

  static BoundaryDofLayout compact(const MixedFESpace& space,
                                   const std::string&  physical_name);

  Index numFacets() const;

  Index numDofs() const;

  const Mesh::BoundaryFacet& facet(Index ib) const;

  Index meshFacetIndex(Index ib) const;

  void facetDofs(Index ib, Vector<Index>& dofs) const;

private:
  BoundaryDofLayout() = default;

  template <typename Space>
  static BoundaryDofLayout compactByTag(const Space& space,
                                        Index        physical_tag)
  {
    BoundaryDofLayout layout;
    layout.mesh_ = &space.mesh();

    std::map<Index, Index> compact_dofs;
    layout.buildByTag(physical_tag,
                      [&space, &compact_dofs](
                          const Mesh::BoundaryFacet& facet,
                          Vector<Index>&             dofs)
                      {
                        appendCompactFacetDofs(
                            space, facet, compact_dofs, dofs);
                      });
    layout.num_dofs_ = static_cast<Index>(compact_dofs.size());
    return layout;
  }

  template <typename Space>
  static BoundaryDofLayout compactByName(const Space&       space,
                                         const std::string& physical_name)
  {
    BoundaryDofLayout layout;
    layout.mesh_ = &space.mesh();

    std::map<Index, Index> compact_dofs;
    layout.buildByName(physical_name,
                       [&space, &compact_dofs](
                           const Mesh::BoundaryFacet& facet,
                           Vector<Index>&             dofs)
                       {
                         appendCompactFacetDofs(
                             space, facet, compact_dofs, dofs);
                       });
    layout.num_dofs_ = static_cast<Index>(compact_dofs.size());
    return layout;
  }

  template <typename DofAppender>
  void buildByTag(Index physical_tag, DofAppender append_dofs)
  {
    const auto& facets = mesh().boundaryFacets();
    for (Index i = 0; i < static_cast<Index>(facets.size()); ++i)
    {
      if (facets[static_cast<std::size_t>(i)].physical_tag == physical_tag)
      {
        addFacet(i, facets[static_cast<std::size_t>(i)], append_dofs);
      }
    }
    if (facet_dofs_.empty())
    {
      throw std::runtime_error(
          "No boundary facets found for physical tag "
          + std::to_string(physical_tag));
    }
  }

  template <typename DofAppender>
  void buildByName(const std::string& physical_name, DofAppender append_dofs)
  {
    const auto& facets = mesh().boundaryFacets();
    for (Index i = 0; i < static_cast<Index>(facets.size()); ++i)
    {
      if (facets[static_cast<std::size_t>(i)].physical_name == physical_name)
      {
        addFacet(i, facets[static_cast<std::size_t>(i)], append_dofs);
      }
    }
    if (facet_dofs_.empty())
    {
      throw std::runtime_error(
          "No boundary facets found for physical name " + physical_name);
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
  const Mesh*                mesh_{nullptr};
  Index                      num_dofs_{0};
  Vector<Index>              facet_indices_;
  IndexSetList               facet_dofs_;
};

} // namespace assembly
} // namespace femx
