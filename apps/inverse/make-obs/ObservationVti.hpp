#pragma once

#include <string>

#include "Config.hpp"
#include <femx/inverse/TimeObservationData.hpp>

namespace femx::make_obs
{

std::string writeObservationVtiOutputs(
    const Params&                       prm,
    const inverse::TimeObservationData& data);

} // namespace femx::make_obs
