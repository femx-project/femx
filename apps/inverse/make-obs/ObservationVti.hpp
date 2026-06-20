#pragma once

#include <string>

#include "Config.hpp"
#include <femx/problem/TimeObservationData.hpp>

namespace femx::make_obs
{

std::string writeObservationVtiOutputs(
    const Params&                       prm,
    const problem::TimeObservationData& data);

} // namespace femx::make_obs
