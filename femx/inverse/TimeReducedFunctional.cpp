#include "TimeObjectivePlan.hpp"
#include <femx/inverse/TimeReducedFunctional.hpp>

namespace femx::inverse::detail
{

class ObjectiveEval<MemorySpace::Device>::Impl final
{
public:
  TimeObjectivePlan plan;
};

ObjectiveEval<MemorySpace::Device>::ObjectiveEval(
    const TimeObjective& obj,
    CudaContext&         ctx)
  : impl_(std::make_unique<Impl>())
{
  impl_->plan.add(obj, ctx);
}

ObjectiveEval<MemorySpace::Device>::~ObjectiveEval() = default;

Index ObjectiveEval<MemorySpace::Device>::numSteps() const noexcept
{
  return impl_->plan.numSteps();
}

Index ObjectiveEval<MemorySpace::Device>::numStates() const noexcept
{
  return impl_->plan.numStates();
}

Index ObjectiveEval<MemorySpace::Device>::numParams() const noexcept
{
  return impl_->plan.numParams();
}

Real ObjectiveEval<MemorySpace::Device>::value(
    const state::DeviceTimeTrajectory& tr,
    const DeviceVector&                prm,
    CudaContext&                       ctx) const
{
  return impl_->plan.value(tr, prm.view(), ctx);
}

void ObjectiveEval<MemorySpace::Device>::stateGrad(
    Index                              level,
    const state::DeviceTimeTrajectory& tr,
    const DeviceVector&                prm,
    DeviceVector&                      out,
    CudaContext&                       ctx) const
{
  impl_->plan.stateGrad(level, tr, prm.view(), out.view(), ctx);
}

void ObjectiveEval<MemorySpace::Device>::paramGrad(
    const state::DeviceTimeTrajectory& tr,
    const DeviceVector&                prm,
    DeviceVector&                      out,
    CudaContext&                       ctx) const
{
  impl_->plan.paramGrad(tr, prm.view(), out.view(), ctx);
}

} // namespace femx::inverse::detail
