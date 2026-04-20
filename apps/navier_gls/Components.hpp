#pragma once

#include <array>
#include <vector>

#include "NavierGLS.hpp"
#include <refem/forms/integrators/DomainBilinearIntegrator.hpp>
#include <refem/forms/integrators/DomainLinearIntegrator.hpp>

namespace refem
{

struct QPState
{
  std::array<real_type, dim> u{};
  std::array<real_type, dim> u_adv{};
  real_type                  grad[dim][dim]{};
  std::array<real_type, dim> grad_u_adv{};
  std::array<real_type, dim> tau{};
};

class TransientLHS final : public DomainBilinearIntegrator
{
public:
  void assemble(const ElementValues& ev,
                DenseMatrix&         Ke) const override;
};

class TransientRHS final : public DomainLinearIntegrator
{
public:
  explicit TransientRHS(const std::vector<QPState>& state);

  void assemble(const ElementValues& ev,
                Vector&              Fe) const override;

private:
  const std::vector<QPState>& state_;
};

class AdvectionLHS final : public DomainBilinearIntegrator
{
public:
  explicit AdvectionLHS(const std::vector<QPState>& state);

  void assemble(const ElementValues& ev,
                DenseMatrix&         Ke) const override;

private:
  const std::vector<QPState>& state_;
};

class AdvectionRHS final : public DomainLinearIntegrator
{
public:
  explicit AdvectionRHS(const std::vector<QPState>& state);

  void assemble(const ElementValues& ev,
                Vector&              Fe) const override;

private:
  const std::vector<QPState>& state_;
};

class ViscousLHS final : public DomainBilinearIntegrator
{
public:
  void assemble(const ElementValues& ev,
                DenseMatrix&         Ke) const override;
};

class ViscousRHS final : public DomainLinearIntegrator
{
public:
  explicit ViscousRHS(const std::vector<QPState>& state);

  void assemble(const ElementValues& ev,
                Vector&              Fe) const override;

private:
  const std::vector<QPState>& state_;
};

class PressureVelocityCouplingLHS final : public DomainBilinearIntegrator
{
public:
  void assemble(const ElementValues& ev,
                DenseMatrix&         Ke) const override;
};

class StabilizationLHS final : public DomainBilinearIntegrator
{
public:
  explicit StabilizationLHS(const std::vector<QPState>& state);

  void assemble(const ElementValues& ev,
                DenseMatrix&         Ke) const override;

private:
  const std::vector<QPState>& state_;
};

class StabilizationRHS final : public DomainLinearIntegrator
{
public:
  explicit StabilizationRHS(const std::vector<QPState>& state);

  void assemble(const ElementValues& ev,
                Vector&              Fe) const override;

private:
  const std::vector<QPState>& state_;
};

} // namespace refem
