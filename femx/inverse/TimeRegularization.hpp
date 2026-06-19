#pragma once

#include <femx/common/Types.hpp>
#include <femx/eq/TimeStateTrajectory.hpp>
#include <femx/inverse/TimeObjectiveFunctional.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx
{
namespace inverse
{

/**
 * @brief Quadratic parameter regularization for time-blocked controls.
 *
 * Parameters are ordered as consecutive time levels, each with block_size
 * entries. The value is
 *
 *   beta_value / 2 * ||m - m_ref||^2
 * + beta_difference / 2 * sum_l ||(m_l - m_ref_l)
 *                              - (m_{l-1} - m_ref_{l-1})||^2.
 */
class TimeRegularization final : public TimeObjectiveFunctional
{
public:
  TimeRegularization(Index               num_steps,
                     Index               num_states,
                     Index               num_levels,
                     Index               block_size,
                     Real                beta_difference,
                     Real                beta_value = 0.0,
                     const Vector<Real>& reference  = {});

  Index numSteps() const override;

  Index numStates() const override;

  Index numParams() const override;

  Real value(const eq::TimeStateTrajectory& tr,
             const Vector<Real>&            prm) const override;

  void stateGrad(Index                          level,
                 const eq::TimeStateTrajectory& tr,
                 const Vector<Real>&            prm,
                 Vector<Real>&                  out) const override;

  void paramGrad(const eq::TimeStateTrajectory& tr,
                 const Vector<Real>&            prm,
                 Vector<Real>&                  out) const override;

private:
  Index index(Index level,
              Index component) const;

  Real centered(const Vector<Real>& prm,
                Index               level,
                Index               component) const;

  void checkParamSize(const Vector<Real>& prm) const;

  static void resize(Vector<Real>& out,
                     Index         size);

private:
  Index        num_steps_{0};
  Index        num_states_{0};
  Index        num_levels_{0};
  Index        block_size_{0};
  Real         beta_difference_{0.0};
  Real         beta_value_{0.0};
  Vector<Real> reference_;
};

} // namespace inverse
} // namespace femx
