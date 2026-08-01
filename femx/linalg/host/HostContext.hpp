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
  using Base = Context<MemorySpace::Host>;

public:
  /**
   * @copydoc Base::vectorHandler()
   */
  VectorHandler<MemorySpace::Host>& vectorHandler() noexcept override
  {
    return vec_handler_;
  }

  /**
   * @copydoc Base::matrixHandler()
   */
  MatrixHandler<MemorySpace::Host>& matrixHandler() noexcept override
  {
    return mat_handler_;
  }

  /**
   * @copydoc Base::elementRange()
   *
   * @details Assigns the full range to the serial Host context.
   */
  IndexRange elementRange(Index count) const override
  {
    require(count >= 0,
            "HostContext element count must be nonnegative");
    return {0, count};
  }

  /**
   * @copydoc Base::ownsElement()
   *
   * @details Owns every valid element in serial execution.
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
   * @copydoc Base::allReduceSum()
   *
   * @details Leaves values unchanged in serial execution.
   */
  void allReduceSum(HostVectorView<Real> vals) const override
  {
    static_cast<void>(vals);
  }

  /**
   * @copydoc Base::sync()
   *
   * @details Serial Host execution is synchronous.
   */
  void sync() const override
  {
  }

private:
  HostMatrixHandler mat_handler_; ///< Owned Host matrix operations.
  HostVectorHandler vec_handler_; ///< Owned Host vector operations.
};

} // namespace femx::linalg
