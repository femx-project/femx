#pragma once

#include <string>

#include <femx/common/Math.hpp>
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
   * @brief Scalar or vector point field to write with points.
   *
   * PointField references external data laid out as one value per point and
   * component.
   */
  struct PointField
  {
    std::string         name;               ///< VTK field name.
    Index               num_components = 1; ///< Number of components per point.
    const Vector<Real>* vals{nullptr};      ///< Field values.
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

  /**
   * @brief Write a point cloud as vertex cells with optional point fields.
   *
   * @param[in] fname - Output `.vtu` file name.
   * @param[in] points - Point coordinates.
   * @param[in] fields - Point data fields with points.size() entries per
   * component.
   */
  void writePointCloud(const std::string&        fname,
                       const Vector<Point3>&     points,
                       const Vector<PointField>& fields) const;
};

} // namespace femx
