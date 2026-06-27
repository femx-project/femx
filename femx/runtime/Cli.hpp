#pragma once

#include <string>

namespace femx::runtime
{

std::string requireValue(int                argc,
                         char**             argv,
                         int&               index,
                         const std::string& key);

} // namespace femx::runtime
