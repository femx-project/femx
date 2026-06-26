#pragma once

#include <iosfwd>
#include <string>

#include <femx/common/Types.hpp>
#include <femx/linalg/DenseMatrix.hpp>
#include <femx/linalg/Vector.hpp>

namespace femx::inverse
{

struct ElemRange
{
  Index begin = 0;
  Index end   = 0;
};

enum class VectorFileHeader
{
  none,
  size
};

std::string requireValue(
    int                argc,
    char**             argv,
    int&               i,
    const std::string& key);

void ensureParentDir(const std::string& path);

void writeVector(
    const std::string&  path,
    const Vector<Real>& x,
    VectorFileHeader    header = VectorFileHeader::none);

void writeMatrix(
    const std::string& path,
    const DenseMatrix& mat);

} // namespace femx::inverse
