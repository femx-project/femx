#pragma once

#include <femx/common/Checks.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/host/HostMatrixHandler.hpp>
#include <femx/linalg/host/HostVectorHandler.hpp>

namespace femx::linalg
{

/**
 * @brief Provide serial Host execution resources.
 */
class HostContext final : public Context<MemorySpace::Host>
{
public:
  /**
   * @brief Return the owned Host vector operations.
   */
  VectorHandler<MemorySpace::Host>& vectorHandler() noexcept override
  {
    return vec_handler_;
  }

  /**
   * @brief Return the owned Host matrix operations.
   */
  MatrixHandler<MemorySpace::Host>& matrixHandler() noexcept override
  {
    return mat_handler_;
  }

  /**
   * @brief Return the full element range.
   *
   * @param[in] count - Element count.
   * @return Full half-open element range.
   * @throws - If `count` is negative.
   */
  IndexRange elementRange(Index count) const override
  {
    require(count >= 0,
            "HostContext element count must be nonnegative");
    return {0, count};
  }

  /**
   * @brief Report ownership of every valid serial Host element.
   *
   * @param[in] element - Global element index.
   * @param[in] count - Global element count.
   * @param[in] rows - Element rows, which may be empty.
   * @return `true` for every valid element.
   * @throws - If `element` or `count` is invalid.
   */
  bool ownsElement(
      Index                       element,
      Index                       count,
      HostVectorView<const Index> rows) const override
  {
    static_cast<void>(rows);
    require(count >= 0 && element >= 0 && element < count,
            "HostContext element index is out of range");
    return true;
  }

  /**
   * @brief Leave serial Host values unchanged.
   *
   * @param[in,out] vals - Values left unchanged.
   */
  void allReduceSum(HostVectorView<Real> vals) const override
  {
    static_cast<void>(vals);
  }

  /**
   * @brief Complete pending Host work; serial execution is synchronous.
   */
  void sync() const override
  {
  }

private:
  HostMatrixHandler mat_handler_; ///< Owned Host matrix operations.
  HostVectorHandler vec_handler_; ///< Owned Host vector operations.
};

} // namespace femx::linalg
