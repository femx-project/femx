#pragma once

#include <algorithm>

#include <femx/common/Checks.hpp>
#include <femx/common/Vector.hpp>

namespace femx::linalg
{

/** @brief Provide Host vector operations. */
class HostVectorHandler final
{
public:
  /**
   * @brief Copy between same-sized Host views.
   *
   * @param[in] src - Source view.
   * @param[out] dst - Destination view.
   * @throws std::runtime_error - If sizes differ or views partially overlap.
   */
  template <class T>
  void copy(VectorView<MemorySpace::Host, const T> src,
            VectorView<MemorySpace::Host, T>       dst) const
  {
    require(src.size() == dst.size(),
            "Host view copy requires equal sizes");
    if (src.empty() || src.data() == dst.data())
    {
      return;
    }
    require(!femx::detail::overlaps(src, dst),
            "Host view copy does not support partial overlap");
    std::copy(src.begin(), src.end(), dst.begin());
  }

  /**
   * @brief Copy between same-sized mutable Host views.
   *
   * @param[in] src - Source view.
   * @param[out] dst - Destination view.
   * @throws std::runtime_error - If sizes differ or views partially overlap.
   */
  template <class T>
  void copy(VectorView<MemorySpace::Host, T> src,
            VectorView<MemorySpace::Host, T> dst) const
  {
    copy(VectorView<MemorySpace::Host, const T>(src.data(), src.size()),
         dst);
  }

  /**
   * @brief Replace a Host vector by copying a view.
   *
   * @param[in] src - Source view.
   * @param[out] dst - Destination vector.
   * @throws std::runtime_error - If the view size is negative.
   */
  template <class T>
  void copy(VectorView<MemorySpace::Host, const T> src,
            Vector<MemorySpace::Host, T>&          dst) const
  {
    dst = src;
  }

  /**
   * @brief Replace a Host vector by copying a mutable view.
   *
   * @param[in] src - Source view.
   * @param[out] dst - Destination vector.
   * @throws std::runtime_error - If the view size is negative.
   */
  template <class T>
  void copy(VectorView<MemorySpace::Host, T> src,
            Vector<MemorySpace::Host, T>&    dst) const
  {
    copy(VectorView<MemorySpace::Host, const T>(src.data(), src.size()),
         dst);
  }

  /**
   * @brief Replace a Host vector with copies of one value.
   *
   * @param[out] out - Vector to replace.
   * @param[in] size - Required vector size.
   * @param[in] val - Value assigned to every entry.
   * @throws std::runtime_error - If `size` is negative.
   */
  template <class T>
  void assign(
      Vector<MemorySpace::Host, T>&                            out,
      Index                                                    size,
      const typename Vector<MemorySpace::Host, T>::value_type& val) const
  {
    out.assign(size, val);
  }

  /**
   * @brief Set every value to zero.
   *
   * @param[out] vals - Values to clear.
   */
  void zero(HostVectorView<Real> vals) const;

  /**
   * @brief Compute `y = a * x + b * y`.
   *
   * @param[in] a - Input-vector scale.
   * @param[in] x - Input vector.
   * @param[in] b - Existing-output scale.
   * @param[in,out] y - Output vector.
   * @throws std::runtime_error - If sizes or storage overlap are invalid.
   */
  void axpby(Real                       a,
             HostVectorView<const Real> x,
             Real                       b,
             HostVectorView<Real>       y) const;

  /**
   * @brief Compute the dot product of two vectors.
   *
   * @param[in] x - First input vector.
   * @param[in] y - Second input vector.
   * @return Dot product of `x` and `y`.
   * @throws std::runtime_error - If vector sizes differ.
   */
  Real dot(HostVectorView<const Real> x, HostVectorView<const Real> y) const;

  /**
   * @brief Compute the squared Euclidean norm of a vector.
   *
   * @param[in] x - Input vector.
   * @return Squared Euclidean norm of `x`.
   */
  Real squaredNorm(HostVectorView<const Real> x) const;

  /**
   * @brief Gather indexed source values into a contiguous destination.
   *
   * @param[in] src - Source values.
   * @param[in] indices - Source indices in destination order.
   * @param[out] dst - Contiguous destination values.
   * @throws std::runtime_error - If sizes, indices, or aliasing are invalid.
   */
  void gather(HostVectorView<const Real>  src,
              HostVectorView<const Index> indices,
              HostVectorView<Real>        dst) const;

  /**
   * @brief Scatter contiguous source values to indexed destinations.
   *
   * @param[in] src - Contiguous source values.
   * @param[in] indices - Destination indices in source order.
   * @param[out] dst - Indexed destination values.
   * @throws std::runtime_error - If sizes, indices, or aliasing are invalid.
   */
  void scatter(HostVectorView<const Real>  src,
               HostVectorView<const Index> indices,
               HostVectorView<Real>        dst) const;
};

} // namespace femx::linalg
