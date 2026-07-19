#pragma once

#include <femx/common/Context.hpp>
#include <femx/common/LinearInterpolation.hpp>
#include <femx/common/Types.hpp>
#include <femx/inverse/TimeObjective.hpp>
#include <femx/inverse/TimeObservationOperator.hpp>
#include <femx/linalg/CsrMatrix.hpp>
#include <femx/state/TimeTrajectory.hpp>

namespace femx
{
namespace inverse
{

class TimeLeastSquaresObjective;
class TimeRegularization;
class TimeBlockRegularization;

/**
 * @brief Device-native aggregate of built-in time objective terms.
 *
 * add() flattens Host objective metadata and performs all required H2D copies.
 * Evaluation then reuses owned Device data and work vectors. Only value()
 * transfers one scalar back to Host; gradient operations remain asynchronous.
 * Device observation operators are explicitly copied and owned by this object.
 * One instance is not reentrant and must be evaluated by one thread at a time.
 */
class TimeObjectivePlan
{
public:
  TimeObjectivePlan() = default;

  TimeObjectivePlan(const TimeObjectivePlan&)                = delete;
  TimeObjectivePlan& operator=(const TimeObjectivePlan&)     = delete;
  TimeObjectivePlan(TimeObjectivePlan&&) noexcept            = default;
  TimeObjectivePlan& operator=(TimeObjectivePlan&&) noexcept = default;

  /**
   * @brief Recursively add a built-in objective and copy its data to Device.
   *
   * Supported terms are TimeLeastSquaresObjective, TimeRegularization,
   * TimeBlockRegularization, and nested SumTimeObjective. A least-squares
   * observation must provide a state-linear, parameter-independent
   * TimeObservationOperator::copyToDevice(). The call synchronizes `ctx` once
   * after all one-time metadata and observation uploads have been enqueued.
   */
  void add(const TimeObjective& obj, CudaContext& ctx);

  /** @brief Number of residual steps expected by this aggregate. */
  Index numSteps() const noexcept;

  /** @brief State-vector size expected at every time level. */
  Index numStates() const noexcept;

  /** @brief Parameter-vector size expected by regularization terms. */
  Index numParams() const noexcept;

  /**
   * @brief Evaluate all terms and return the sole Device-to-Host scalar.
   *
   * No allocation or H2D transfer occurs. `tr` and `prm` remain on Device.
   */
  Real value(const state::DeviceTimeTrajectory& tr,
             DeviceConstVectorView              prm,
             CudaContext&                       ctx) const;

  /**
   * @brief Overwrite `out` with the state gradient at one time level.
   *
   * The operation is enqueued on `ctx` and does not allocate or transfer to
   * Host. `out` must contain numStates() entries.
   */
  void stateGrad(Index                              level,
                 const state::DeviceTimeTrajectory& tr,
                 DeviceConstVectorView              prm,
                 DeviceVectorView                   out,
                 CudaContext&                       ctx) const;

  /**
   * @brief Overwrite `out` with the full parameter gradient on Device.
   *
   * The operation is enqueued on `ctx` and reuses the flattened quadratic
   * data created by add().
   */
  void paramGrad(const state::DeviceTimeTrajectory& tr,
                 DeviceConstVectorView              prm,
                 DeviceVectorView                   out,
                 CudaContext&                       ctx) const;

private:
  struct LeastSquaresTerm
  {
    std::unique_ptr<DeviceTimeObservationOperator> obs;
    Array<LinearInterpolation>                     interp;
    HostVector                                     row_wts;
    HostVector                                     data_h;
    HostVector                                     obs_wts_h;
    DeviceVector                                   data;
    DeviceVector                                   obs_wts;
    Index                                          num_obs{0};
  };

  void flatten(const TimeObjective& obj, CudaContext& ctx);
  void addLeastSquares(
      const TimeLeastSquaresObjective&               obj,
      std::unique_ptr<DeviceTimeObservationOperator> obs);
  void uploadLeastSquares(Index first, CudaContext& ctx);
  void addRegularization(const TimeRegularization& obj);
  void addRegularization(const TimeBlockRegularization& obj);
  void appendQuadratic(Index row,
                       Index col,
                       Real  val,
                       Real  row_ref,
                       Real  col_ref);
  void setDims(const TimeObjective& obj);
  void uploadQuadratic(CudaContext& ctx);
  void checkInputs(const state::DeviceTimeTrajectory& tr,
                   DeviceConstVectorView              prm) const;
  void checkLevel(Index level) const;

private:
  Index num_steps_{-1};
  Index num_states_{-1};
  Index num_prm_{-1};

  Array<LeastSquaresTerm> ls_;

  HostIndexVector q_rows_h_;
  HostIndexVector q_cols_h_;
  HostVector      q_vals_h_;
  HostVector      q_lin_h_;
  Real            q_const_{0.0};
  Index           q_terms_{0};

  DeviceCsrMatrix      q_mat_;
  DeviceVector         q_lin_;
  mutable DeviceVector q_prod_;
  mutable DeviceVector q_dot_;

  mutable DeviceVector lo_;
  mutable DeviceVector hi_;
  mutable DeviceVector dir_;
  mutable DeviceVector scalar_;
  mutable HostVector   scalar_h_;
};

} // namespace inverse
} // namespace femx
