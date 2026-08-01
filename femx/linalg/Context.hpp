#pragma once

#include <femx/common/Types.hpp>
#include <femx/common/View.hpp>
#include <femx/linalg/MatrixHandler.hpp>
#include <femx/linalg/VectorHandler.hpp>

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
  virtual VectorHandler<MemorySpace::Host>& vectorHandler() noexcept = 0;

  /**
   * @brief Return the owned Host matrix operations.
   */
  virtual MatrixHandler<MemorySpace::Host>& matrixHandler() noexcept = 0;

  /**
   * @brief Return the elements assigned to this context.
   *
   * @param[in] count - Global element count.
   * @return Assigned half-open element range.
   * @throws std::runtime_error If validation fails.
   */
  virtual IndexRange elementRange(Index count) const = 0;

  /**
   * @brief Report whether this context owns an element contribution.
   *
   * @param[in] element - Global element index.
   * @param[in] count   - Global element count.
   * @param[in] rows    - Element residual rows in application numbering.
   * @return `true` when this context should evaluate the element.
   * @throws std::runtime_error If validation fails.
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
   * @throws std::runtime_error If validation fails.
   */
  virtual void allReduceSum(HostVectorView<Real> vals) const = 0;

  /**
   * @brief Wait for work submitted to this context.
   *
   * @throws std::runtime_error If validation fails.
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
   * @brief Return the owned Device vector operations.
   */
  virtual VectorHandler<MemorySpace::Device>& vectorHandler() noexcept = 0;

  /**
   * @brief Return the owned Device matrix operations.
   */
  virtual MatrixHandler<MemorySpace::Device>& matrixHandler() noexcept = 0;

  /**
   * @brief Return the elements assigned to this context.
   *
   * @param[in] count - Global element count.
   * @return Assigned half-open element range.
   * @throws std::runtime_error If validation fails.
   */
  virtual IndexRange elementRange(Index count) const = 0;

  /**
   * @brief Wait for work submitted to this context.
   *
   * @throws std::runtime_error If validation fails.
   */
  virtual void sync() const = 0;
};

} // namespace femx::linalg
