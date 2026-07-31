#pragma once

#include <femx/common/Types.hpp>
#include <femx/common/View.hpp>
#include <femx/linalg/cuda/CudaVectorHandler.hpp>
#include <femx/linalg/native/HostVectorHandler.hpp>

namespace femx::linalg
{

/**
 * @brief Represent a half-open element range.
 */
struct IndexRange
{
  Index begin{0}; ///< First element index.
  Index end{0};   ///< One-past-the-last element index.
};

/**
 * @brief Define execution resources for a memory space.
 */
template <MemorySpace Space>
class Context;

/**
 * @brief Define Host execution resources and collective operations.
 */
template <>
class Context<MemorySpace::Host>
{
public:
  virtual ~Context() = default;

  /**
   * @brief Return the owned Host vector operations.
   */
  virtual HostVectorHandler& vectorHandler() noexcept = 0;

  /**
   * @brief Return the elements assigned to this context.
   *
   * @param[in] count - Global element count.
   * @return Assigned half-open element range.
   * @throws - If `count` is negative or the execution
   * environment is invalid.
   */
  virtual IndexRange elementRange(Index count) const = 0;

  /**
   * @brief Report whether this context owns an element contribution.
   *
   * @param[in] element - Global element index.
   * @param[in] count - Global element count.
   * @param[in] rows - Element residual rows in application numbering.
   * @return `true` when this context should evaluate the element.
   * @throws - If the element data or execution environment
   * is invalid.
   */
  virtual bool ownsElement(
      Index                       element,
      Index                       count,
      HostVectorView<const Index> rows) const
  {
    static_cast<void>(rows);
    const IndexRange range = elementRange(count);
    return element >= range.begin && element < range.end;
  }

  /**
   * @brief Sum replicated Host values across the execution context.
   *
   * @param[in,out] vals - Values replaced by their global sums.
   * @throws - If a collective operation fails.
   */
  virtual void allReduceSum(HostVectorView<Real> vals) const = 0;

  /**
   * @brief Wait for work submitted to this context.
   *
   * @throws - If synchronization fails.
   */
  virtual void sync() const = 0;
};

/**
 * @brief Define Device execution resources.
 */
template <>
class Context<MemorySpace::Device>
{
public:
  virtual ~Context() = default;

  /**
   * @brief Return the owned CUDA vector operations.
   */
  virtual CudaVectorHandler& vectorHandler() noexcept = 0;

  /**
   * @brief Return the elements assigned to this context.
   *
   * @param[in] count - Global element count.
   * @return Assigned half-open element range.
   * @throws - If `count` is negative.
   */
  virtual IndexRange elementRange(Index count) const = 0;

  /**
   * @brief Wait for work submitted to this context.
   *
   * @throws - If synchronization fails.
   */
  virtual void sync() const = 0;
};

} // namespace femx::linalg
