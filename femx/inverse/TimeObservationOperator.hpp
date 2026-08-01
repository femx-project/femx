#pragma once

#include <memory>

#include <femx/common/Types.hpp>
#include <femx/common/View.hpp>

namespace femx::linalg
{
template <MemorySpace Space>
class Context;
} // namespace femx::linalg

namespace femx
{
namespace inverse
{

/**
 * @brief Device-resident, state-linear, parameter-independent observation.
 *
 * Implementations own Device-ready observation data. All vector arguments are
 * non-owning Device views, and operations must not allocate or mirror storage.
 */
class DeviceTimeObservationOperator
{
public:
  using ConstView = DeviceVectorView<const Real>;
  using View      = DeviceVectorView<Real>;

  virtual ~DeviceTimeObservationOperator() = default;

  /**
   * @brief Return the number of residual time steps.
   */
  virtual Index numSteps() const        = 0;
  /**
   * @brief Return the state size at one time level.
   */
  virtual Index numStates() const       = 0;
  /**
   * @brief Return the observation size at one time level.
   */
  virtual Index numObservations() const = 0;

  /**
   * @brief Overwrite a preallocated observation vector on `ctx`.
   */
  virtual void observe(Index                                 level,
                       ConstView                             state,
                       View                                  out,
                       linalg::Context<MemorySpace::Device>& ctx) const = 0;

  /**
   * @brief Add the state-transpose product to preallocated `out` on `ctx`.
   */
  virtual void addStateJacT(Index                                 level,
                            ConstView                             dir,
                            View                                  out,
                            linalg::Context<MemorySpace::Device>& ctx) const = 0;
};

/**
 * @brief Observation map y_l = H_l(u_l, m) for time objectives.
 *
 * Implementations evaluate observations and provide Jacobian and
 * transpose-Jacobian products with respect to state and parameter variables.
 */
class TimeObservationOperator
{
public:
  virtual ~TimeObservationOperator() = default;

  /**
   * @brief Return the number of residual time steps.
   */
  virtual Index numSteps() const        = 0;
  /**
   * @brief Return the state size at one time level.
   */
  virtual Index numStates() const       = 0;
  /**
   * @brief Return the full parameter-vector size.
   */
  virtual Index numParams() const       = 0;
  /**
   * @brief Return the observation size at one time level.
   */
  virtual Index numObservations() const = 0;

  /**
   * @brief Create and explicitly copy an independently owned Device operator.
   *
   * The default implementation reports that Device execution is unsupported.
   * Implementations enqueue any required copies on `ctx`; the caller owns the
   * returned operator and controls when initialization is synchronized.
   */
  virtual std::unique_ptr<DeviceTimeObservationOperator> copyToDevice(
      linalg::Context<MemorySpace::Device>&) const
  {
    return {};
  }

  virtual void observe(Index                   level,
                       const HostVector<Real>& state,
                       const HostVector<Real>& prm,
                       HostVector<Real>&       out) const = 0;

  virtual void applyStateJac(Index                   level,
                             const HostVector<Real>& state,
                             const HostVector<Real>& prm,
                             const HostVector<Real>& dir,
                             HostVector<Real>&       out) const = 0;

  virtual void applyStateJacT(Index                   level,
                              const HostVector<Real>& state,
                              const HostVector<Real>& prm,
                              const HostVector<Real>& dir,
                              HostVector<Real>&       out) const = 0;

  virtual void applyParamJac(Index                   level,
                             const HostVector<Real>& state,
                             const HostVector<Real>& prm,
                             const HostVector<Real>& dir,
                             HostVector<Real>&       out) const = 0;

  virtual void applyParamJacT(Index                   level,
                              const HostVector<Real>& state,
                              const HostVector<Real>& prm,
                              const HostVector<Real>& dir,
                              HostVector<Real>&       out) const = 0;
};

} // namespace inverse
} // namespace femx
