#pragma once

#include <string>

#include <femx/common/Types.hpp>

namespace femx
{

class Mesh;

template <typename T>
class Vector;

/**
 * @brief Write unstructured meshes and point data in VTK XML VTU format.
 *
 * VtuWriter writes binary XML `.vtu` files for visualization in tools such as
 * ParaView.  It is intentionally lightweight and does not require HDF5.
 */
class VtuWriter
{
public:
  /**
   * @brief Scalar or vector point field to write with the mesh.
   *
   * PointField references external data laid out as one value per mesh point
   * and component.
   */
  struct PointField
  {
    std::string         name;
    Index               num_components = 1;
    const Vector<Real>* vals{nullptr};
  };

  /**
   * @brief Write mesh geometry, connectivity, and point fields.
   *
   * @param[in] fname - Output `.vtu` file name.
   * @param[in] mesh - Mesh to write.
   * @param[in] fields - Point data fields with mesh.numNodes() entries per
   * component.
   */
  void writePointData(const std::string&        fname,
                      const Mesh&               mesh,
                      const Vector<PointField>& fields) const;
};

} // namespace femx
