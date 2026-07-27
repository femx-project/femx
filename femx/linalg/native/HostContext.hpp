#pragma once

#include <femx/common/Checks.hpp>
#include <femx/linalg/Context.hpp>
#include <femx/linalg/native/HostVectorHandler.hpp>

namespace femx::linalg
{

/** @brief Provide serial Host execution resources. */
class HostContext final : public Context<MemorySpace::Host>
{
public:
  /** @brief Return the owned Host vector operations. */
  HostVectorHandler& vectors() noexcept override
  {
    return vec_handler_;
  }

  /**
   * @brief Return the full element range.
   *
   * @param[in] count - Element count.
   * @return Full half-open element range.
   * @throws std::runtime_error - If `count` is negative.
   */
  IndexRange elementRange(Index count) const override
  {
    require(count >= 0,
            "HostContext element count must be nonnegative");
    return {0, count};
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

  /** @brief Complete pending Host work; serial execution is synchronous. */
  void sync() const override
  {
  }

private:
  HostVectorHandler vec_handler_; ///< Owned Host vector operations.
};

} // namespace femx::linalg
